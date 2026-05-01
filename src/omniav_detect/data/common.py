"""
本文件功能：
- 提供 FakeAVCeleb 和 MAVOS-DD 转换为 ms-swift / Qwen2.5-Omni SFT JSONL 的公共工具。

主要内容：
- scan_video_files：扫描数据集目录并统计视频文件。
- make_binary_record / make_structured_record：生成 ms-swift 对话格式样本。
- write_stats：写出扫描摘要、统计文件、缺失/异常文件报告和预览样本。

使用方式：
- 被 `omniav_detect.data.fakeavceleb` 和 `omniav_detect.data.mavosdd` 复用。
"""

from __future__ import annotations

import json
import logging
import random
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

Sample = Dict[str, Any]
Record = Dict[str, Any]
Issue = Dict[str, Any]

SYSTEM_PROMPT = (
    "You are an audio-video deepfake detector. Given an input video, answer only Real or Fake."
)
BINARY_USER_PROMPT = (
    "<video>\nGiven the video, please assess if it's Real or Fake? Only answer Real or Fake."
)
STRUCTURED_USER_PROMPT = (
    "<video>\nGiven the video, please assess if it's Real or Fake? Return a compact JSON object "
    "with overall_label, video_label, audio_label, modality_type, and evidence."
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def normalize_json_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def abs_path(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False))


def safe_relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        return path.name


def compact_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not metadata:
        return {}
    return {str(key): normalize_json_value(value) for key, value in metadata.items()}


def record_issue(
    issues: List[Issue],
    dataset: str,
    split: str,
    reason: str,
    expected_path: Path | str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    issues.append(
        {
            "dataset": dataset,
            "split": split,
            "reason": reason,
            "expected_path": str(expected_path),
            "metadata_json": json.dumps(compact_metadata(metadata), ensure_ascii=False, sort_keys=True),
        }
    )


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            children = list(current.iterdir())
        except OSError as exc:
            logging.warning("Could not list %s: %s", current, exc)
            continue
        for child in children:
            if child.is_dir():
                stack.append(child)
            elif child.is_file():
                yield child


def scan_video_files(root: Path) -> Dict[str, Any]:
    """
    函数功能：
    - 扫描一个数据集根目录，统计一级目录、文件扩展名、视频数量和总大小。

    参数：
    - root: 数据集根目录。

    返回：
    - 字典，包含 root 是否存在、一级目录、扩展名计数、视频总数和视频总大小。

    关键逻辑：
    - 递归遍历真实文件，只把受支持的视频扩展名计入视频总数。
    """
    summary = {
        "root": abs_path(root),
        "exists": root.exists(),
        "first_level_dirs": [],
        "extension_counts": {},
        "total_video_files": 0,
        "total_size_bytes": 0,
    }
    if not root.exists():
        return summary

    try:
        summary["first_level_dirs"] = sorted(child.name for child in root.iterdir() if child.is_dir())
    except OSError as exc:
        logging.warning("Could not list first-level directories under %s: %s", root, exc)

    extension_counts: Counter[str] = Counter()
    total_video_files = 0
    total_size_bytes = 0
    for file_path in iter_files(root):
        suffix = file_path.suffix.lower() or "<no_ext>"
        extension_counts[suffix] += 1
        if suffix in SUPPORTED_VIDEO_EXTENSIONS:
            total_video_files += 1
            try:
                total_size_bytes += file_path.stat().st_size
            except OSError:
                pass

    summary["extension_counts"] = dict(sorted(extension_counts.items()))
    summary["total_video_files"] = total_video_files
    summary["total_size_bytes"] = total_size_bytes
    return summary


def first_present(row: Dict[str, Any], names: Iterable[str]) -> str:
    lowered = {str(key).lower(): key for key in row}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            value = clean_text(row.get(key))
            if value:
                return value
    return ""


def make_sample(video_path: Path, meta: Dict[str, Any]) -> Sample:
    return {"video_path": abs_path(video_path), "meta": meta}


def limit_samples_per_class(
    samples: List[Sample], max_samples_per_class: Optional[int], seed: int
) -> List[Sample]:
    if max_samples_per_class is None:
        return samples
    if max_samples_per_class < 1:
        return []

    grouped: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample["meta"].get("overall_label", "Unknown")].append(sample)

    rng = random.Random(seed)
    limited: List[Sample] = []
    for label in sorted(grouped):
        items = list(grouped[label])
        rng.shuffle(items)
        limited.extend(items[:max_samples_per_class])
    limited.sort(key=lambda item: item["video_path"])
    return limited


def make_messages(user_prompt: str, assistant_content: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_content},
    ]


def make_binary_record(sample: Sample) -> Record:
    """
    函数功能：
    - 将内部样本转换为 ms-swift binary SFT 记录。

    参数：
    - sample: 包含 video_path 和 meta 的内部样本。

    返回：
    - 包含 messages、videos 和 meta 的 JSONL 单行记录。

    关键逻辑：
    - assistant 只输出 `Real` 或 `Fake`，视频路径保持绝对路径。
    """
    meta = sample["meta"]
    return {
        "messages": make_messages(BINARY_USER_PROMPT, meta["overall_label"]),
        "videos": [sample["video_path"]],
        "meta": meta,
    }


def make_structured_record(sample: Sample) -> Record:
    """
    函数功能：
    - 将内部样本转换为 structured SFT 记录。

    参数：
    - sample: 包含 video_path 和 meta 的内部样本。

    返回：
    - assistant content 为 JSON 字符串的 ms-swift 记录。

    关键逻辑：
    - content 字段仍是字符串，保证整行 JSONL 可解析。
    """
    meta = sample["meta"]
    payload = {
        "overall_label": meta.get("overall_label", "Unknown"),
        "video_label": meta.get("video_label", "Unknown"),
        "audio_label": meta.get("audio_label", "Unknown"),
        "modality_type": meta.get("modality_type", "Unknown"),
        "evidence": make_structured_evidence(meta),
    }
    return {
        "messages": make_messages(
            STRUCTURED_USER_PROMPT,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        ),
        "videos": [sample["video_path"]],
        "meta": meta,
    }


def make_structured_evidence(meta: Dict[str, Any]) -> str:
    video = str(meta.get("video_label", "Unknown")).lower()
    audio = str(meta.get("audio_label", "Unknown")).lower()
    if video in {"real", "fake"} and audio in {"real", "fake"}:
        return f"The video modality is {video} while the audio modality is {audio}."
    return "Only the overall real/fake label is available for this dataset."


def write_jsonl(path: Path, records: Sequence[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False))
            handle.write("\n")
    logging.info("Wrote %d records to %s", len(records), path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    logging.info("Wrote %s", path)


def write_missing_or_invalid(path: Path, issues: Sequence[Issue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "split", "reason", "expected_path", "metadata_json"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for issue in issues:
            writer.writerow({field: issue.get(field, "") for field in fields})
    logging.info("Wrote %d missing/invalid rows to %s", len(issues), path)


def count_meta(metas: Iterable[Dict[str, Any]], key: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for meta in metas:
        value = clean_text(meta.get(key)) or "Unknown"
        counter[value] += 1
    return counter


def build_stats(outputs: Dict[str, List[Record]], issues: Sequence[Issue]) -> Dict[str, Any]:
    output_stats: Dict[str, Any] = {}
    for filename, records in outputs.items():
        metas = [record.get("meta", {}) for record in records]
        stat = {
            "sample_count": len(records),
            "label_distribution": dict(count_meta(metas, "overall_label")),
        }
        if filename.startswith("fakeavceleb"):
            stat["modality_type_distribution"] = dict(count_meta(metas, "modality_type"))
        if filename.startswith("mavosdd"):
            stat["language_distribution"] = dict(count_meta(metas, "language"))
            stat["generative_method_distribution"] = dict(count_meta(metas, "generative_method"))
        output_stats[filename] = stat

    reason_counts = Counter(issue["reason"] for issue in issues)
    missing_file_count = sum(
        count
        for reason, count in reason_counts.items()
        if reason in {"missing_file", "missing_root", "missing_category_dir", "missing_video_path_field"}
    )
    invalid_file_count = sum(
        count
        for reason, count in reason_counts.items()
        if reason not in {"missing_file", "missing_root", "missing_category_dir", "missing_video_path_field"}
    )
    return {
        "outputs": output_stats,
        "missing_file_count": missing_file_count,
        "invalid_file_count": invalid_file_count,
        "missing_or_invalid_by_reason": dict(sorted(reason_counts.items())),
    }


def build_preview_samples(
    outputs: Dict[str, List[Record]], num_preview: int, seed: int
) -> Dict[str, List[Record]]:
    rng = random.Random(seed)
    preview: Dict[str, List[Record]] = {}
    for filename, records in outputs.items():
        if len(records) <= num_preview:
            preview[filename] = list(records)
        else:
            preview[filename] = rng.sample(list(records), num_preview)
    return preview


def write_stats(
    output_dir: Path,
    scan_summary: Dict[str, Any],
    outputs: Dict[str, List[Record]],
    issues: Sequence[Issue],
    num_preview: int,
    seed: int,
) -> None:
    """
    函数功能：
    - 写出数据准备阶段的扫描、统计、缺失异常文件和预览样本。

    参数：
    - output_dir: 输出目录。
    - scan_summary: 数据集扫描摘要。
    - outputs: 各 JSONL 输出文件对应的记录。
    - issues: 缺失或异常文件记录。
    - num_preview: 每个输出文件保存的预览样本数。
    - seed: 随机预览采样种子。

    返回：
    - 无返回值，直接写文件。

    关键逻辑：
    - 所有统计文件写到同一个 output_dir，便于人工检查数据准备质量。
    """
    write_json(output_dir / "dataset_scan_summary.json", scan_summary)
    write_json(output_dir / "dataset_stats.json", build_stats(outputs, issues))
    write_missing_or_invalid(output_dir / "missing_or_invalid_files.csv", issues)
    write_json(output_dir / "preview_samples.json", build_preview_samples(outputs, num_preview, seed))


def write_output_jsonl(output_dir: Path, outputs: Dict[str, List[Record]], dry_run: bool) -> None:
    if dry_run:
        logging.info("Dry run enabled: dataset jsonl files will not be written.")
        return
    for filename, records in outputs.items():
        write_jsonl(output_dir / filename, records)
