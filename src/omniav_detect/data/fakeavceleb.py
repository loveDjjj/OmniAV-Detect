"""
本文件功能：
- 负责扫描 FakeAVCeleb 本地视频并生成 ms-swift / Qwen2.5-Omni SFT JSONL。

主要内容：
- build_fakeavceleb_samples：从四个 FakeAVCeleb 模态目录构建有效样本。
- stratified_split：按 overall_label + modality_type 分层切分 train/eval。
- build_fakeavceleb_fold_splits：按 MRDF 的 subject-independent 5 折文件生成 train/test。
- build_fakeavceleb_output_records：生成默认 train/eval 输出记录。
- build_fakeavceleb_fold_output_records：生成 5 折 protocol 输出记录。

使用方式：
- 被统一入口 `omniav_detect.data.prepare_runner` 调用。
"""

from __future__ import annotations

import csv
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from omniav_detect.data.common import (
    Issue,
    SUPPORTED_VIDEO_EXTENSIONS,
    Sample,
    abs_path,
    clean_text,
    first_present,
    iter_files,
    limit_samples_per_class,
    make_sample,
    make_binary_record,
    make_structured_record,
    normalize_json_value,
    record_issue,
    safe_relative_path,
)


FAKEAVCELEB_CATEGORIES = {
    "RealVideo-RealAudio": {
        "overall_label": "Real",
        "video_label": "Real",
        "audio_label": "Real",
        "modality_type": "R-R",
    },
    "RealVideo-FakeAudio": {
        "overall_label": "Fake",
        "video_label": "Real",
        "audio_label": "Fake",
        "modality_type": "R-F",
    },
    "FakeVideo-RealAudio": {
        "overall_label": "Fake",
        "video_label": "Fake",
        "audio_label": "Real",
        "modality_type": "F-R",
    },
    "FakeVideo-FakeAudio": {
        "overall_label": "Fake",
        "video_label": "Fake",
        "audio_label": "Fake",
        "modality_type": "F-F",
    },
}


def read_fakeavceleb_metadata(root: Path) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    metadata_path = root / "meta_data.csv"
    if not metadata_path.exists():
        return {}, []

    try:
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [
                {str(key): normalize_json_value(value) for key, value in row.items() if key}
                for row in reader
            ]
    except Exception as exc:
        logging.warning("Could not read FakeAVCeleb metadata from %s: %s", metadata_path, exc)
        return {}, []

    index: Dict[str, Dict[str, Any]] = {}
    ambiguous: set[str] = set()
    for row in rows:
        for key in fakeavceleb_metadata_keys(row, root):
            if key in ambiguous:
                continue
            if key in index and index[key] != row:
                index.pop(key, None)
                ambiguous.add(key)
                continue
            index[key] = row

    logging.info(
        "Loaded %d FakeAVCeleb metadata rows and built %d lookup keys from %s",
        len(rows),
        len(index),
        metadata_path,
    )
    return index, rows


def fakeavceleb_metadata_keys(row: Dict[str, Any], root: Path) -> Iterable[str]:
    for candidate in fakeavceleb_metadata_path_candidates(row, root):
        yield abs_path(candidate)
        yield candidate.resolve(strict=False).as_posix()
        yield safe_relative_path(candidate, root)
        yield candidate.name
        yield candidate.stem


def fakeavceleb_metadata_path_candidates(row: Dict[str, Any], root: Path) -> Iterable[Path]:
    path_columns = [
        key
        for key in row
        if any(token in key.lower() for token in ("path", "file", "filename", "video"))
    ]
    for column in path_columns:
        value = clean_text(row.get(column))
        if not value:
            continue
        has_path_shape = "/" in value or "\\" in value or Path(value).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        if not has_path_shape:
            continue
        raw_path = Path(value)
        if raw_path.is_absolute():
            yield raw_path
            continue
        structured_path = fakeavceleb_structured_metadata_path(row, raw_path, root)
        if structured_path is not None:
            yield structured_path
            continue
        yield root / raw_path


def fakeavceleb_structured_metadata_path(
    row: Dict[str, Any], raw_path: Path, root: Path
) -> Optional[Path]:
    if raw_path.parent != Path("."):
        return None
    category = clean_text(row.get("type"))
    race = clean_text(row.get("race"))
    gender = clean_text(row.get("gender"))
    source = clean_text(row.get("source"))
    if not all([category, race, gender, source]):
        return None
    if category not in FAKEAVCELEB_CATEGORIES:
        return None
    return root / category / race / gender / source / raw_path


def fakeavceleb_expected_paths_from_metadata(row: Dict[str, Any], root: Path) -> Iterable[Path]:
    yield from fakeavceleb_metadata_path_candidates(row, root)


def record_fakeavceleb_metadata_missing_files(
    root: Path,
    metadata_rows: Sequence[Dict[str, Any]],
    missing_or_invalid: List[Issue],
) -> None:
    emitted: set[str] = set()
    for row in metadata_rows:
        for expected_path in fakeavceleb_expected_paths_from_metadata(row, root):
            if expected_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
                continue
            absolute = abs_path(expected_path)
            if absolute in emitted:
                continue
            emitted.add(absolute)
            if expected_path.exists():
                continue
            split = first_present(row, ["split", "original_split", "partition"]) or "metadata"
            record_issue(missing_or_invalid, "FakeAVCeleb", split, "missing_file", expected_path, row)


def find_fakeavceleb_metadata(
    path: Path, root: Path, metadata_index: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if not metadata_index:
        return None
    candidates = [
        abs_path(path),
        path.resolve(strict=False).as_posix(),
        safe_relative_path(path, root),
        path.name,
        path.stem,
    ]
    for key in candidates:
        if key in metadata_index:
            return metadata_index[key]
    return None


def merge_optional_metadata(meta: Dict[str, Any], source_metadata: Optional[Dict[str, Any]]) -> None:
    if not source_metadata:
        return

    language = first_present(
        source_metadata,
        ["language", "lang", "open_set_language_name", "target_language"],
    )
    method = first_present(
        source_metadata,
        ["generative_method", "generation_method", "method", "model", "open_set_model_name"],
    )
    split = first_present(source_metadata, ["split", "original_split", "partition"])
    if language and not meta.get("language"):
        meta["language"] = language
    if method and not meta.get("generative_method"):
        meta["generative_method"] = method
    if split and not meta.get("original_split"):
        meta["original_split"] = split
    meta["source_metadata"] = {str(key): normalize_json_value(value) for key, value in source_metadata.items()}


def fakeavceleb_subject_id(path: Path, root: Path, metadata: Optional[Dict[str, Any]]) -> str:
    """
    函数功能：
    - 解析 FakeAVCeleb 样本所属的 subject id，优先使用 metadata 的 `source` 字段。

    参数：
    - path: 当前视频路径。
    - root: FakeAVCeleb 根目录。
    - metadata: 可选 metadata 行。

    返回：
    - 形如 `id00076` 的 subject id；无法解析时返回空字符串。
    """
    if metadata is not None:
        source = clean_text(metadata.get("source"))
        if source:
            return source

    try:
        relative_parts = path.resolve(strict=False).relative_to(root.resolve(strict=False)).parts
    except ValueError:
        relative_parts = path.parts
    for part in reversed(relative_parts[:-1]):
        text = clean_text(part)
        if text.lower().startswith("id"):
            return text
    return ""


def load_fakeavceleb_fold_ids(folds_root: Path) -> Dict[int, Dict[str, set[str]]]:
    """
    函数功能：
    - 读取 MRDF 风格的 `train_*.txt` / `test_*.txt` 5 折划分文件。

    参数：
    - folds_root: 存放 5 折 txt 文件的目录。

    返回：
    - `fold_index -> {"train": set(...), "test": set(...)}` 映射。

    关键逻辑：
    - 文件内容按 subject id 逐行读取，自动去除空白和重复。
    """
    folds: Dict[int, Dict[str, set[str]]] = {}
    for fold_idx in range(1, 6):
        fold_data: Dict[str, set[str]] = {}
        for split_name in ("train", "test"):
            split_path = folds_root / f"{split_name}_{fold_idx}.txt"
            if not split_path.exists():
                raise ValueError(f"Missing FakeAVCeleb fold file: {split_path}")
            values = {
                clean_text(line)
                for line in split_path.read_text(encoding="utf-8-sig").splitlines()
                if clean_text(line)
            }
            if not values:
                raise ValueError(f"FakeAVCeleb fold file is empty: {split_path}")
            fold_data[split_name] = values
        overlap = fold_data["train"] & fold_data["test"]
        if overlap:
            raise ValueError(
                f"FakeAVCeleb fold {fold_idx} has overlapping subject ids in train/test: {sorted(overlap)[:10]}"
            )
        folds[fold_idx] = fold_data
    return folds


def build_fakeavceleb_samples(
    root: Path,
    max_samples_per_class: Optional[int],
    seed: int,
    missing_or_invalid: List[Issue],
) -> List[Sample]:
    """
    函数功能：
    - 扫描 FakeAVCeleb 四类目录，构建本地真实存在且可用的视频样本。

    参数：
    - root: FakeAVCeleb 根目录。
    - max_samples_per_class: 每个 Real/Fake 类最多保留的样本数，None 表示不限制。
    - seed: 类内限量抽样的随机种子。
    - missing_or_invalid: 用于累积缺失、空文件、非法扩展名等问题。

    返回：
    - 有效样本列表，每个样本包含绝对视频路径和训练所需 meta。

    关键逻辑：
    - 优先扫描本地真实文件；metadata 只用于补充信息和报告缺失文件。
    """
    if not root.exists():
        logging.warning("FakeAVCeleb root does not exist: %s", root)
        record_issue(missing_or_invalid, "FakeAVCeleb", "scan", "missing_root", root)
        return []

    metadata_index, metadata_rows = read_fakeavceleb_metadata(root)
    samples: List[Sample] = []
    for dirname, labels in FAKEAVCELEB_CATEGORIES.items():
        category_dir = root / dirname
        if not category_dir.exists():
            record_issue(
                missing_or_invalid,
                "FakeAVCeleb",
                "scan",
                "missing_category_dir",
                category_dir,
                {"category": dirname},
            )
            logging.warning("FakeAVCeleb category directory is missing: %s", category_dir)
            continue

        for file_path in iter_files(category_dir):
            metadata = find_fakeavceleb_metadata(file_path, root, metadata_index)
            if file_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
                record_issue(
                    missing_or_invalid,
                    "FakeAVCeleb",
                    "scan",
                    "invalid_extension",
                    file_path,
                    metadata,
                )
                continue
            try:
                size = file_path.stat().st_size
            except OSError as exc:
                record_issue(
                    missing_or_invalid,
                    "FakeAVCeleb",
                    "scan",
                    f"stat_error:{exc.__class__.__name__}",
                    file_path,
                    metadata,
                )
                continue
            if size <= 0:
                record_issue(
                    missing_or_invalid,
                    "FakeAVCeleb",
                    "scan",
                    "zero_size_file",
                    file_path,
                    metadata,
                )
                continue

            meta = {
                "dataset": "FakeAVCeleb",
                "source_path": abs_path(file_path),
                "overall_label": labels["overall_label"],
                "video_label": labels["video_label"],
                "audio_label": labels["audio_label"],
                "modality_type": labels["modality_type"],
                "language": "",
                "generative_method": "",
                "original_split": "",
            }
            merge_optional_metadata(meta, metadata)
            subject_id = fakeavceleb_subject_id(file_path, root, metadata)
            if subject_id:
                meta["subject_id"] = subject_id
            samples.append(make_sample(file_path, meta))

    record_fakeavceleb_metadata_missing_files(root, metadata_rows, missing_or_invalid)
    samples = limit_samples_per_class(samples, max_samples_per_class, seed)
    logging.info("Built %d valid FakeAVCeleb samples from %s", len(samples), root)
    return samples


def stratified_split(
    samples: List[Sample], train_ratio: float, seed: int
) -> Tuple[List[Sample], List[Sample]]:
    """
    函数功能：
    - 对 FakeAVCeleb 样本做分层 train/eval 切分。

    参数：
    - samples: 有效样本列表。
    - train_ratio: 训练集比例。
    - seed: 分层打乱随机种子。

    返回：
    - train 样本列表和 eval 样本列表。

    关键逻辑：
    - 分层键为 `(overall_label, modality_type)`，避免不同模态组合分布严重漂移。
    """
    grouped: Dict[Tuple[str, str], List[Sample]] = defaultdict(list)
    for sample in samples:
        meta = sample.get("meta", {})
        key = (meta.get("overall_label", "Unknown"), meta.get("modality_type", "Unknown"))
        grouped[key].append(sample)

    rng = random.Random(seed)
    train: List[Sample] = []
    eval_samples: List[Sample] = []
    for key in sorted(grouped):
        group = list(grouped[key])
        rng.shuffle(group)
        if len(group) == 1:
            train_count = 1
        else:
            train_count = int(len(group) * train_ratio)
            train_count = max(1, min(len(group) - 1, train_count))
        train.extend(group[:train_count])
        eval_samples.extend(group[train_count:])

    train.sort(key=lambda item: item.get("video_path", ""))
    eval_samples.sort(key=lambda item: item.get("video_path", ""))
    return train, eval_samples


def build_fakeavceleb_fold_splits(
    samples: List[Sample],
    folds_root: Path,
) -> Dict[int, Dict[str, List[Sample]]]:
    """
    函数功能：
    - 按 MRDF 的 subject-independent 5 折文件，把样本拆成 train/test。

    参数：
    - samples: 已扫描得到的 FakeAVCeleb 样本列表。
    - folds_root: `train_*.txt` / `test_*.txt` 所在目录。

    返回：
    - `fold_index -> {"train": [...], "test": [...]}`。

    关键逻辑：
    - 以样本的 `meta.subject_id` 匹配划分文件；未命中任何 fold 的样本会报错，避免静默漏数。
    """
    folds = load_fakeavceleb_fold_ids(folds_root)
    splits: Dict[int, Dict[str, List[Sample]]] = {
        fold_idx: {"train": [], "test": []} for fold_idx in folds
    }
    unmatched_subjects: set[str] = set()
    missing_subject_paths: List[str] = []

    for sample in samples:
        meta = sample.get("meta", {})
        subject_id = clean_text(meta.get("subject_id"))
        if not subject_id:
            missing_subject_paths.append(sample.get("video_path", ""))
            continue
        matched_any_fold = False
        for fold_idx, fold_data in folds.items():
            if subject_id in fold_data["train"]:
                splits[fold_idx]["train"].append(sample)
                matched_any_fold = True
            elif subject_id in fold_data["test"]:
                splits[fold_idx]["test"].append(sample)
                matched_any_fold = True
        if not matched_any_fold:
            unmatched_subjects.add(subject_id)

    if missing_subject_paths:
        preview = missing_subject_paths[:5]
        raise ValueError(
            "Some FakeAVCeleb samples have no subject_id, so MRDF 5-fold split cannot be applied: "
            f"{preview}"
        )
    if unmatched_subjects:
        raise ValueError(
            "Some FakeAVCeleb subject ids are not covered by the MRDF 5-fold files: "
            f"{sorted(unmatched_subjects)[:20]}"
        )

    for fold_idx, fold_splits in splits.items():
        for split_name in ("train", "test"):
            fold_splits[split_name].sort(key=lambda item: item.get("video_path", ""))
    return splits


def build_fakeavceleb_output_records(samples: List[Sample], args: Any) -> Dict[str, List[Dict[str, Any]]]:
    outputs: Dict[str, List[Dict[str, Any]]] = {}
    train_samples, eval_samples = stratified_split(samples, args.fakeavceleb_train_ratio, args.seed)

    if args.mode in {"binary", "both"}:
        outputs["fakeavceleb_binary_train.jsonl"] = [make_binary_record(sample) for sample in train_samples]
        outputs["fakeavceleb_binary_eval.jsonl"] = [make_binary_record(sample) for sample in eval_samples]

    if args.mode in {"structured", "both"}:
        outputs["fakeavceleb_structured_train.jsonl"] = [
            make_structured_record(sample) for sample in train_samples
        ]
        outputs["fakeavceleb_structured_eval.jsonl"] = [
            make_structured_record(sample) for sample in eval_samples
        ]

    return outputs


def build_fakeavceleb_fold_output_records(samples: List[Sample], args: Any) -> Dict[str, List[Dict[str, Any]]]:
    """
    函数功能：
    - 按 MRDF 的 5 折 protocol 生成 FakeAVCeleb 训练/测试 JSONL。

    参数：
    - samples: 有效样本列表。
    - args: 运行参数，要求包含 `folds_root`、`mode`。

    返回：
    - 输出文件名到记录列表的映射。
    """
    outputs: Dict[str, List[Dict[str, Any]]] = {}
    split_map = build_fakeavceleb_fold_splits(samples, Path(args.folds_root))

    for fold_idx in sorted(split_map):
        train_samples = split_map[fold_idx]["train"]
        test_samples = split_map[fold_idx]["test"]
        if args.mode in {"binary", "both"}:
            outputs[f"fakeavceleb_mrdf5fold_fold{fold_idx}_binary_train.jsonl"] = [
                make_binary_record(sample) for sample in train_samples
            ]
            outputs[f"fakeavceleb_mrdf5fold_fold{fold_idx}_binary_test.jsonl"] = [
                make_binary_record(sample) for sample in test_samples
            ]
        if args.mode in {"structured", "both"}:
            outputs[f"fakeavceleb_mrdf5fold_fold{fold_idx}_structured_train.jsonl"] = [
                make_structured_record(sample) for sample in train_samples
            ]
            outputs[f"fakeavceleb_mrdf5fold_fold{fold_idx}_structured_test.jsonl"] = [
                make_structured_record(sample) for sample in test_samples
            ]
    return outputs
