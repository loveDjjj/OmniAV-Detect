"""FakeAVCeleb-specific conversion logic for ms-swift/Qwen2.5-Omni SFT."""

from __future__ import annotations

import csv
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from prepare_swift_av_sft_common import (
    Issue,
    SUPPORTED_VIDEO_EXTENSIONS,
    Sample,
    abs_path,
    clean_text,
    first_present,
    iter_files,
    limit_samples_per_class,
    make_sample,
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
    pathish_columns = [
        key
        for key in row
        if any(token in key.lower() for token in ("path", "file", "video", "filename", "name"))
    ]
    for column in pathish_columns:
        value = clean_text(row.get(column))
        if not value:
            continue
        raw_path = Path(value)
        yield value.replace("\\", "/")
        yield raw_path.name
        yield raw_path.stem
        if raw_path.is_absolute():
            yield abs_path(raw_path)
        else:
            yield (root / raw_path).resolve(strict=False).as_posix()


def fakeavceleb_expected_paths_from_metadata(row: Dict[str, Any], root: Path) -> Iterable[Path]:
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
        yield raw_path if raw_path.is_absolute() else root / raw_path


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


def build_fakeavceleb_samples(
    root: Path,
    max_samples_per_class: Optional[int],
    seed: int,
    missing_or_invalid: List[Issue],
) -> List[Sample]:
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
            samples.append(make_sample(file_path, meta))

    record_fakeavceleb_metadata_missing_files(root, metadata_rows, missing_or_invalid)
    samples = limit_samples_per_class(samples, max_samples_per_class, seed)
    logging.info("Built %d valid FakeAVCeleb samples from %s", len(samples), root)
    return samples


def stratified_split(
    samples: List[Sample], train_ratio: float, seed: int
) -> Tuple[List[Sample], List[Sample]]:
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
