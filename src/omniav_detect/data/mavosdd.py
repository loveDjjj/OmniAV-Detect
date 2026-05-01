"""
本文件功能：
- 负责读取 MAVOS-DD 本地 Arrow 数据与视频文件，生成 ms-swift / Qwen2.5-Omni binary SFT JSONL。

主要内容：
- load_mavos_dataset：通过 datasets.load_from_disk 读取本地数据。
- build_mavosdd_samples：校验视频文件并按官方 split/open-set 字段构建样本。
- build_mavosdd_output_records：生成 MAVOS-DD binary 输出记录。

使用方式：
- 被统一入口 `omniav_detect.data.prepare_runner` 调用。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from omniav_detect.data.common import (
    Issue,
    SUPPORTED_VIDEO_EXTENSIONS,
    Sample,
    abs_path,
    clean_text,
    limit_samples_per_class,
    make_sample,
    make_binary_record,
    normalize_bool,
    record_issue,
)


MAVOS_OUTPUT_SPLITS = (
    "train",
    "validation",
    "test_indomain",
    "test_open_model",
    "test_open_language",
    "test_open_full",
    "test_all",
)


def load_mavos_dataset(root: Path) -> Optional[Any]:
    """
    函数功能：
    - 使用 Hugging Face datasets 从本地 MAVOS-DD 目录读取 Arrow 数据。

    参数：
    - root: MAVOS-DD 根目录。

    返回：
    - datasets.Dataset 或 DatasetDict；读取失败时返回 None。

    关键逻辑：
    - `datasets` 是可选依赖，缺失时只记录提示，不中断整个脚本导入。
    """
    if not root.exists():
        logging.warning("MAVOS-DD root does not exist: %s", root)
        return None
    try:
        from datasets import Dataset, load_from_disk  # type: ignore
    except ImportError:
        logging.warning(
            "The 'datasets' package is required to read MAVOS-DD Arrow data. "
            "Install it with: pip install datasets"
        )
        return None

    try:
        return load_from_disk(str(root))
    except Exception as first_error:
        try:
            return Dataset.load_from_disk(str(root))
        except Exception as second_error:
            logging.warning(
                "Could not load MAVOS-DD with datasets.load_from_disk(%s): %s; "
                "Dataset.load_from_disk also failed: %s",
                root,
                first_error,
                second_error,
            )
            return None


def iter_mavos_rows(dataset: Any) -> Iterable[Tuple[Dict[str, Any], str]]:
    if dataset is None:
        return
    if hasattr(dataset, "items"):
        for split_name, split_dataset in dataset.items():
            for row in split_dataset:
                yield dict(row), str(split_name)
    else:
        for row in dataset:
            yield dict(row), ""


def resolve_mavos_video_path(root: Path, value: Any) -> Path:
    if isinstance(value, dict):
        value = value.get("path") or value.get("filename") or value.get("file") or ""
    text = clean_text(value)
    if not text:
        return root / ""
    path = Path(text)
    if path.is_absolute():
        return path
    return root / path


def build_mavosdd_samples(
    root: Path,
    max_samples_per_class: Optional[int],
    seed: int,
    missing_or_invalid: List[Issue],
) -> Dict[str, List[Sample]]:
    """
    函数功能：
    - 从 MAVOS-DD metadata 构建 train/validation/test/open-set 各输出 split 的有效样本。

    参数：
    - root: MAVOS-DD 根目录。
    - max_samples_per_class: 每个 Real/Fake 类最多保留的样本数，None 表示不限制。
    - seed: 类内限量抽样的随机种子。
    - missing_or_invalid: 用于累积缺失和异常文件记录。

    返回：
    - split 名称到样本列表的映射。

    关键逻辑：
    - 只使用 metadata 中的 overall real/fake 标签，不伪造音频/视频单模态标签。
    """
    split_samples = {split: [] for split in MAVOS_OUTPUT_SPLITS}
    if not root.exists():
        record_issue(missing_or_invalid, "MAVOS-DD", "scan", "missing_root", root)
        return split_samples

    dataset = load_mavos_dataset(root)
    if dataset is None:
        return split_samples

    total_rows = 0
    for row, dataset_split in iter_mavos_rows(dataset):
        total_rows += 1
        split = clean_text(row.get("split")) or dataset_split
        split = split.lower()
        expected_path = resolve_mavos_video_path(root, row.get("video_path"))
        issue_split = split or "metadata"

        if not clean_text(row.get("video_path")):
            record_issue(missing_or_invalid, "MAVOS-DD", issue_split, "missing_video_path_field", root, row)
            continue
        if expected_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            record_issue(
                missing_or_invalid,
                "MAVOS-DD",
                issue_split,
                "invalid_extension",
                expected_path,
                row,
            )
            continue
        if not expected_path.exists():
            record_issue(missing_or_invalid, "MAVOS-DD", issue_split, "missing_file", expected_path, row)
            continue
        try:
            size = expected_path.stat().st_size
        except OSError as exc:
            record_issue(
                missing_or_invalid,
                "MAVOS-DD",
                issue_split,
                f"stat_error:{exc.__class__.__name__}",
                expected_path,
                row,
            )
            continue
        if size <= 0:
            record_issue(missing_or_invalid, "MAVOS-DD", issue_split, "zero_size_file", expected_path, row)
            continue

        label = clean_text(row.get("label")).lower()
        overall_label = "Real" if label == "real" else "Fake"
        meta = {
            "dataset": "MAVOS-DD",
            "source_path": abs_path(expected_path),
            "overall_label": overall_label,
            "video_label": "Unknown",
            "audio_label": "Unknown",
            "modality_type": "Unknown",
            "language": mavos_language(row),
            "generative_method": mavos_generative_method(row),
            "original_split": split,
        }
        sample = make_sample(expected_path, meta)
        for split_name in mavos_output_memberships(split, row):
            split_samples[split_name].append(sample)

    for split_name, samples in list(split_samples.items()):
        split_samples[split_name] = limit_samples_per_class(samples, max_samples_per_class, seed)
    logging.info(
        "Loaded %d MAVOS-DD metadata rows and built %d split-assigned valid samples",
        total_rows,
        sum(len(samples) for samples in split_samples.values()),
    )
    return split_samples


def mavos_language(row: Dict[str, Any]) -> str:
    for key in ("language", "lang", "target_language"):
        value = clean_text(row.get(key))
        if value:
            return value
    return ""


def mavos_generative_method(row: Dict[str, Any]) -> str:
    for key in ("generative_method", "generation_method", "method", "model", "source_model"):
        value = clean_text(row.get(key))
        if value and value.lower() not in {"true", "false"}:
            return value
    return ""


def mavos_output_memberships(split: str, row: Dict[str, Any]) -> List[str]:
    normalized = split.lower()
    if normalized in {"train", "training"}:
        return ["train"]
    if normalized in {"validation", "val", "dev"}:
        return ["validation"]
    if normalized != "test":
        return []

    open_model = normalize_bool(row.get("open_set_model"))
    open_language = normalize_bool(row.get("open_set_language"))
    memberships = ["test_all"]
    if not open_model and not open_language:
        memberships.append("test_indomain")
    elif open_model and not open_language:
        memberships.append("test_open_model")
    elif not open_model and open_language:
        memberships.append("test_open_language")
    else:
        memberships.append("test_open_full")
    return memberships


def build_mavosdd_output_records(split_samples: Dict[str, List[Sample]]) -> Dict[str, List[Dict[str, Any]]]:
    outputs: Dict[str, List[Dict[str, Any]]] = {}
    for split_name in MAVOS_OUTPUT_SPLITS:
        outputs[f"mavosdd_binary_{split_name}.jsonl"] = [
            make_binary_record(sample) for sample in split_samples[split_name]
        ]
    return outputs
