#!/usr/bin/env python3
"""Prepare FakeAVCeleb and MAVOS-DD videos for ms-swift/Qwen2.5-Omni SFT."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from prepare_fakeavceleb_swift_sft import (
    FAKEAVCELEB_CATEGORIES,
    build_fakeavceleb_samples,
    stratified_split,
)
from prepare_mavosdd_swift_sft import MAVOS_OUTPUT_SPLITS, build_mavosdd_samples
from prepare_swift_av_sft_common import Issue, Record, Sample, clean_text, scan_video_files


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
