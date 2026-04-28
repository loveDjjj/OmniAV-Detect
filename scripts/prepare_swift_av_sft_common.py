"""Shared helpers for ms-swift audio-video SFT dataset preparation."""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

Sample = Dict[str, Any]
Record = Dict[str, Any]
Issue = Dict[str, Any]


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
