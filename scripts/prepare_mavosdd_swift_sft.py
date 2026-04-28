"""MAVOS-DD-specific conversion logic for ms-swift/Qwen2.5-Omni SFT."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from prepare_swift_av_sft_common import (
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
    scan_video_files,
    setup_logging,
    write_output_jsonl,
    write_stats,
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


def build_mavosdd_output_records(split_samples: Dict[str, List[Sample]]) -> Dict[str, List[Dict[str, Any]]]:
    outputs: Dict[str, List[Dict[str, Any]]] = {}
    for split_name in MAVOS_OUTPUT_SPLITS:
        outputs[f"mavosdd_binary_{split_name}.jsonl"] = [
            make_binary_record(sample) for sample in split_samples[split_name]
        ]
    return outputs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert local MAVOS-DD videos to ms-swift/Qwen2.5-Omni binary SFT jsonl."
    )
    parser.add_argument("--mavos_root", default="/data/OneDay/MAVOS-DD")
    parser.add_argument("--output_dir", default="/data/OneDay/OmniAV-Detect/data/swift_sft/mavosdd")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples_per_class", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--num_preview", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_logging()
    args = parse_args(argv)
    root = Path(args.mavos_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_samples_per_class is not None and args.max_samples_per_class < 1:
        logging.error("--max_samples_per_class must be positive when provided")
        return 2

    logging.info("Preparing MAVOS-DD binary SFT data, dry_run=%s", args.dry_run)
    logging.info("MAVOS-DD root: %s", root)
    logging.info("Output dir: %s", output_dir)

    issues: List[Issue] = []
    scan_summary = {"MAVOS-DD": scan_video_files(root)}
    split_samples = build_mavosdd_samples(
        root=root,
        max_samples_per_class=args.max_samples_per_class,
        seed=args.seed,
        missing_or_invalid=issues,
    )
    outputs = build_mavosdd_output_records(split_samples)

    write_output_jsonl(output_dir, outputs, args.dry_run)
    write_stats(output_dir, scan_summary, outputs, issues, args.num_preview, args.seed)

    logging.info("Prepared %d MAVOS-DD output jsonl definitions.", len(outputs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
