"""
本文件功能：
- 扫描解压后的 MVAD 视频，构建样本索引，并执行 group-aware train/val 划分。

主要内容：
- build_samples：扫描视频、配对分离音频并构建样本 meta。
- group_aware_split：按 group_id 划分，避免同组样本泄漏。
- write_split_outputs：写出索引、分组 manifest 和统计。
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mvad.common import iter_video_files, parse_video_sample, split_counts, write_json, write_jsonl
from mvad.pairing import attach_audio_pairs


def setup_logging() -> None:
    """初始化日志格式。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def build_samples(unpack_root: Path, require_audio_pair: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    函数功能：
    - 扫描解压目录并构造 MVAD 样本列表。
    - 如果要求显式音频配对，则过滤找不到分离音频的样本并返回缺失报告。
    """
    raw_samples = [parse_video_sample(path, unpack_root) for path in iter_video_files(unpack_root)]
    if not raw_samples:
        raise ValueError(f"No videos found under {unpack_root}")
    samples, missing_audio = attach_audio_pairs(raw_samples, unpack_root, require_audio_pair=require_audio_pair)
    samples.sort(key=lambda item: item["video_path"])
    missing_audio.sort(key=lambda item: item["video_path"])
    return samples, missing_audio


def group_aware_split(
    samples: Sequence[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    函数功能：
    - 按 group_id 划分 train/val，确保同一个 group 不跨 split。

    参数：
    - samples: 样本列表。
    - val_ratio: 验证集 group 比例。
    - seed: 随机种子。

    返回：
    - train 样本和 val 样本。
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[sample["meta"]["group_id"]].append(sample)
    groups = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(groups)
    val_group_count = max(1, round(len(groups) * val_ratio))
    if val_group_count >= len(groups):
        val_group_count = len(groups) - 1
    val_groups = set(groups[:val_group_count])
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for group_id in groups:
        target = val if group_id in val_groups else train
        target.extend(grouped[group_id])
    train.sort(key=lambda item: item["video_path"])
    val.sort(key=lambda item: item["video_path"])
    return train, val


def build_group_manifest(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """构造 group_id 到样本数量和标签信息的 manifest。"""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[sample["meta"]["group_id"]].append(sample)
    rows = []
    for group_id in sorted(grouped):
        group_samples = grouped[group_id]
        first_meta = group_samples[0]["meta"]
        rows.append(
            {
                "group_id": group_id,
                "count": len(group_samples),
                "overall_label": first_meta["overall_label"],
                "modality_type": first_meta["modality_type"],
                "mvad_modality": first_meta["mvad_modality"],
                "examples": [item["meta"]["relative_path"] for item in group_samples[:5]],
            }
        )
    return rows


def split_stats(train: Sequence[Dict[str, Any]], val: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总 train/val 统计信息。"""
    train_groups = {item["meta"]["group_id"] for item in train}
    val_groups = {item["meta"]["group_id"] for item in val}
    overlap = sorted(train_groups & val_groups)
    if overlap:
        raise ValueError(f"Group leakage detected: {overlap[:10]}")
    return {
        "train": split_counts(train),
        "val": split_counts(val),
        "group_overlap_count": len(overlap),
    }


def write_split_outputs(
    samples: Sequence[Dict[str, Any]],
    train: Sequence[Dict[str, Any]],
    val: Sequence[Dict[str, Any]],
    work_root: Path,
    missing_audio: Sequence[Dict[str, Any]] | None = None,
) -> None:
    """写出索引、split 和统计文件。"""
    write_jsonl(work_root / "mvad_all_index.jsonl", samples)
    write_jsonl(work_root / "mvad_train_index.jsonl", train)
    write_jsonl(work_root / "mvad_val_index.jsonl", val)
    write_jsonl(work_root / "missing_audio_pairs.jsonl", missing_audio or [])
    write_jsonl(work_root / "group_manifest.jsonl", build_group_manifest(samples))
    stats = split_stats(train, val)
    stats["missing_audio_pair_count"] = len(missing_audio or [])
    write_json(work_root / "split_stats.json", stats)
    write_json(
        work_root / "split_preview.json",
        {
            "train": [item["meta"]["relative_path"] for item in list(train)[:20]],
            "val": [item["meta"]["relative_path"] for item in list(val)[:20]],
        },
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析索引和划分参数。"""
    parser = argparse.ArgumentParser(description="Build MVAD index and group-aware train/val split.")
    parser.add_argument("--unpack_root", required=True)
    parser.add_argument("--work_root", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require_audio_pair", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """索引和划分 CLI 主流程。"""
    setup_logging()
    args = parse_args(argv)
    samples, missing_audio = build_samples(Path(args.unpack_root), require_audio_pair=args.require_audio_pair)
    train, val = group_aware_split(samples, args.val_ratio, args.seed)
    write_split_outputs(samples, train, val, Path(args.work_root), missing_audio=missing_audio)
    logging.info("Built MVAD split: train=%d val=%d missing_audio=%d", len(train), len(val), len(missing_audio))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
