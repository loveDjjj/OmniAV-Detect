"""
本文件功能：
- 串联 MVAD zip 解压、索引划分、音频抽取和 Qwen JSONL 生成。

主要内容：
- parse_args：解析一体化预处理参数。
- main：按需执行 unzip、split 和 JSONL 构建。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

from mvad.build_av_jsonl import build_split_jsonl
from mvad.build_index_and_split import build_samples, group_aware_split, write_split_outputs
from mvad.common import write_json
from mvad.unzip_archives import unpack_archives


def setup_logging() -> None:
    """初始化日志格式。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析 MVAD 一体化预处理参数。"""
    parser = argparse.ArgumentParser(description="Prepare MVAD public train data for Qwen2.5-Omni SFT.")
    parser.add_argument("--source_root", default="/data/MVAD")
    parser.add_argument("--unpack_root", default="/data/OneDay/OmniAV-Detect/data/mvad_unpacked")
    parser.add_argument("--work_root", default="/data/OneDay/OmniAV-Detect/data/mvad_processed")
    parser.add_argument("--audio_root", default="/data/OneDay/OmniAV-Detect/data/audio_cache/mvad")
    parser.add_argument("--jsonl_root", default="/data/OneDay/OmniAV-Detect/data/swift_sft/mvad")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extractor", default="7z", help="Archive extractor executable, default 7z.")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--audio_channels", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_unzip", action="store_true")
    parser.add_argument("--skip_audio", action="store_true")
    parser.add_argument("--skip_bad_archives", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """MVAD 一体化预处理主流程。"""
    setup_logging()
    args = parse_args(argv)
    source_root = Path(args.source_root)
    unpack_root = Path(args.unpack_root)
    work_root = Path(args.work_root)
    jsonl_root = Path(args.jsonl_root)

    if not args.skip_unzip:
        manifest = unpack_archives(
            source_root,
            unpack_root,
            overwrite=args.overwrite,
            extractor=args.extractor,
            skip_bad_archives=args.skip_bad_archives,
        )
        write_json(work_root / "unpack_manifest.json", manifest)
        failed_archives = [row for row in manifest if row.get("status") == "failed"]
        if failed_archives:
            logging.warning(
                "MVAD unpack finished with %d failed archive(s). See %s for details.",
                len(failed_archives),
                work_root / "unpack_manifest.json",
            )
    samples = build_samples(unpack_root)
    train, val = group_aware_split(samples, val_ratio=args.val_ratio, seed=args.seed)
    write_split_outputs(samples, train, val, work_root)
    build_split_jsonl(
        work_root / "mvad_train_index.jsonl",
        jsonl_root / "mvad_binary_train_with_audio.jsonl",
        Path(args.audio_root),
        args.ffmpeg,
        args.sample_rate,
        args.audio_channels,
        args.overwrite,
        args.dry_run,
        args.skip_audio,
    )
    build_split_jsonl(
        work_root / "mvad_val_index.jsonl",
        jsonl_root / "mvad_binary_val_with_audio.jsonl",
        Path(args.audio_root),
        args.ffmpeg,
        args.sample_rate,
        args.audio_channels,
        args.overwrite,
        args.dry_run,
        args.skip_audio,
    )
    logging.info("Prepared MVAD JSONL under %s", jsonl_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
