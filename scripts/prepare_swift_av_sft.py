#!/usr/bin/env python3
"""Prepare FakeAVCeleb and MAVOS-DD videos for ms-swift/Qwen2.5-Omni SFT."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

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

MAVOS_OUTPUT_SPLITS = (
    "train",
    "validation",
    "test_indomain",
    "test_open_model",
    "test_open_language",
    "test_open_full",
    "test_all",
)


Sample = Dict[str, Any]
Record = Dict[str, Any]
Issue = Dict[str, Any]


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

    first_level_dirs = []
    try:
        first_level_dirs = sorted(child.name for child in root.iterdir() if child.is_dir())
    except OSError as exc:
        logging.warning("Could not list first-level directories under %s: %s", root, exc)
    summary["first_level_dirs"] = first_level_dirs

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
    meta["source_metadata"] = compact_metadata(source_metadata)


def first_present(row: Dict[str, Any], names: Sequence[str]) -> str:
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


def make_messages(user_prompt: str, assistant_content: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_content},
    ]


def make_binary_record(sample: Sample) -> Record:
    meta = sample["meta"]
    return {
        "messages": make_messages(BINARY_USER_PROMPT, meta["overall_label"]),
        "videos": [sample["video_path"]],
        "meta": meta,
    }


def make_structured_record(sample: Sample) -> Record:
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


def load_mavos_dataset(root: Path) -> Optional[Any]:
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


def build_output_records(
    fake_samples: List[Sample],
    mavos_splits: Dict[str, List[Sample]],
    args: argparse.Namespace,
) -> Dict[str, List[Record]]:
    outputs: Dict[str, List[Record]] = {}
    fake_train, fake_eval = stratified_split(fake_samples, args.fakeavceleb_train_ratio, args.seed)

    if args.mode in {"binary", "both"}:
        outputs["fakeavceleb_binary_train.jsonl"] = [make_binary_record(sample) for sample in fake_train]
        outputs["fakeavceleb_binary_eval.jsonl"] = [make_binary_record(sample) for sample in fake_eval]
        for split_name in MAVOS_OUTPUT_SPLITS:
            filename = f"mavosdd_binary_{split_name}.jsonl"
            outputs[filename] = [make_binary_record(sample) for sample in mavos_splits[split_name]]

    if args.mode in {"structured", "both"}:
        outputs["fakeavceleb_structured_train.jsonl"] = [
            make_structured_record(sample) for sample in fake_train
        ]
        outputs["fakeavceleb_structured_eval.jsonl"] = [
            make_structured_record(sample) for sample in fake_eval
        ]

    return outputs


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


def count_meta(metas: Iterable[Dict[str, Any]], key: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for meta in metas:
        value = clean_text(meta.get(key)) or "Unknown"
        counter[value] += 1
    return counter


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
    write_json(output_dir / "dataset_scan_summary.json", scan_summary)
    write_json(output_dir / "dataset_stats.json", build_stats(outputs, issues))
    write_missing_or_invalid(output_dir / "missing_or_invalid_files.csv", issues)
    write_json(output_dir / "preview_samples.json", build_preview_samples(outputs, num_preview, seed))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert local FakeAVCeleb and MAVOS-DD videos to ms-swift/Qwen2.5-Omni SFT jsonl."
    )
    parser.add_argument("--fakeavceleb_root", default="/data/OneDay/FakeAVCeleb")
    parser.add_argument("--mavos_root", default="/data/OneDay/MAVOS-DD")
    parser.add_argument("--output_dir", default="/data/OneDay/OmniAV-Detect/data/swift_sft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fakeavceleb_train_ratio", type=float, default=0.7)
    parser.add_argument("--mode", choices=["binary", "structured", "both"], default="binary")
    parser.add_argument("--max_samples_per_class", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--num_preview", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_logging()
    args = parse_args(argv)
    fake_root = Path(args.fakeavceleb_root).expanduser()
    mavos_root = Path(args.mavos_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not 0 < args.fakeavceleb_train_ratio < 1:
        logging.error("--fakeavceleb_train_ratio must be between 0 and 1")
        return 2
    if args.max_samples_per_class is not None and args.max_samples_per_class < 1:
        logging.error("--max_samples_per_class must be positive when provided")
        return 2

    logging.info("Preparing data with mode=%s, dry_run=%s", args.mode, args.dry_run)
    logging.info("FakeAVCeleb root: %s", fake_root)
    logging.info("MAVOS-DD root: %s", mavos_root)
    logging.info("Output dir: %s", output_dir)

    issues: List[Issue] = []
    scan_summary = {
        "FakeAVCeleb": scan_video_files(fake_root),
        "MAVOS-DD": scan_video_files(mavos_root),
    }

    fake_samples = build_fakeavceleb_samples(
        root=fake_root,
        max_samples_per_class=args.max_samples_per_class,
        seed=args.seed,
        missing_or_invalid=issues,
    )
    mavos_splits = build_mavosdd_samples(
        root=mavos_root,
        max_samples_per_class=args.max_samples_per_class,
        seed=args.seed,
        missing_or_invalid=issues,
    )
    outputs = build_output_records(fake_samples, mavos_splits, args)

    if args.dry_run:
        logging.info("Dry run enabled: dataset jsonl files will not be written.")
    else:
        for filename, records in outputs.items():
            write_jsonl(output_dir / filename, records)

    write_stats(
        output_dir=output_dir,
        scan_summary=scan_summary,
        outputs=outputs,
        issues=issues,
        num_preview=args.num_preview,
        seed=args.seed,
    )

    stats = build_stats(outputs, issues)
    logging.info("Prepared %d output jsonl definitions.", len(outputs))
    for filename, stat in stats["outputs"].items():
        logging.info("%s: %d samples %s", filename, stat["sample_count"], stat["label_distribution"])
    logging.info(
        "Missing files: %d; invalid files: %d",
        stats["missing_file_count"],
        stats["invalid_file_count"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
