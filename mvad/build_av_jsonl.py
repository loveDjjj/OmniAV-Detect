"""
本文件功能：
- 从 MVAD split index 抽取音频，并生成 Qwen2.5-Omni 显式 audios JSONL。

主要内容：
- build_jsonl_records：按 index 中的音频策略生成 JSONL 记录。
- build_split_jsonl：读取 index 并写出 JSONL。
- main：命令行入口。
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from mvad.common import make_binary_audio_record, output_audio_path, write_jsonl
from src.omniav_detect.evaluation.progress import create_progress


def setup_logging() -> None:
    """初始化日志格式。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 样本列表。"""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_audio(ffmpeg: str, video_path: Path, audio_path: Path, sample_rate: int, audio_channels: int, overwrite: bool) -> None:
    """
    函数功能：
    - 调用 ffmpeg 从单个视频抽取音频。
    """
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(audio_channels),
        str(audio_path),
    ]
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path} -> {audio_path}: {completed.stderr.strip() or completed.stdout.strip()}"
        )


def build_jsonl_records(
    samples: Iterable[Dict[str, Any]],
    audio_root: Path,
    ffmpeg: str = "ffmpeg",
    sample_rate: int = 16000,
    audio_channels: int = 1,
    overwrite: bool = False,
    dry_run: bool = False,
    skip_audio: bool = False,
) -> List[Dict[str, Any]]:
    """
    函数功能：
    - 对样本生成 Qwen2.5-Omni JSONL 记录。
    - 优先使用 index 中已经配对好的 `audio_path`；仅当 `audio_handling=extract_from_video`
      时才从视频容器里抽音频。

    参数：
    - samples: split index 中的样本。
    - audio_root: 音频输出根目录。
    - dry_run: true 时只规划音频路径，不调用 ffmpeg。
    - skip_audio: true 时也不调用 ffmpeg，用于已抽取音频后的快速重建。

    返回：
    - JSONL 记录列表。
    """
    sample_list = list(samples)
    records = []
    handling_counts = Counter(sample.get("audio_handling", "extract_from_video") for sample in sample_list)
    logging.info(
        "MVAD audio handling: paired_file=%d extract_from_video=%d unknown=%d",
        handling_counts.get("paired_file", 0),
        handling_counts.get("extract_from_video", 0),
        sum(count for key, count in handling_counts.items() if key not in {"paired_file", "extract_from_video"}),
    )
    extraction_plan: List[tuple[Path, Path]] = []
    build_progress = create_progress(total=len(sample_list), desc="build mvad jsonl", unit="sample")
    try:
        for sample in sample_list:
            video_path = Path(sample["video_path"]).expanduser().resolve(strict=False)
            audio_handling = sample.get("audio_handling", "extract_from_video")
            if sample.get("audio_path") and audio_handling == "paired_file":
                audio_path = Path(sample["audio_path"]).expanduser().resolve(strict=False)
            elif sample.get("audio_path") and audio_handling == "extract_from_video":
                audio_path = Path(sample["audio_path"]).expanduser().resolve(strict=False)
            else:
                audio_path = output_audio_path(video_path, audio_root)
            if audio_handling == "extract_from_video" and not dry_run and not skip_audio and (overwrite or not audio_path.exists()):
                extraction_plan.append((video_path, audio_path))
            records.append(make_binary_audio_record(sample, audio_path))
            build_progress.update(1)
    finally:
        build_progress.close()
    if extraction_plan:
        extract_progress = create_progress(total=len(extraction_plan), desc="extract embedded audio", unit="video")
        try:
            for video_path, audio_path in extraction_plan:
                extract_audio(ffmpeg, video_path, audio_path, sample_rate, audio_channels, overwrite)
                extract_progress.update(1)
        finally:
            extract_progress.close()
    return records


def build_split_jsonl(
    index_path: Path,
    output_jsonl: Path,
    audio_root: Path,
    ffmpeg: str,
    sample_rate: int,
    audio_channels: int,
    overwrite: bool,
    dry_run: bool,
    skip_audio: bool,
) -> List[Dict[str, Any]]:
    """读取 index，生成并写出单个 split JSONL。"""
    records = build_jsonl_records(
        load_jsonl(index_path),
        audio_root=audio_root,
        ffmpeg=ffmpeg,
        sample_rate=sample_rate,
        audio_channels=audio_channels,
        overwrite=overwrite,
        dry_run=dry_run,
        skip_audio=skip_audio,
    )
    write_jsonl(output_jsonl, records)
    return records


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析 JSONL 构建命令行参数。"""
    parser = argparse.ArgumentParser(description="Build MVAD Qwen2.5-Omni JSONL with explicit audios.")
    parser.add_argument("--train_index", required=True)
    parser.add_argument("--val_index", required=True)
    parser.add_argument("--jsonl_root", required=True)
    parser.add_argument("--audio_root", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--audio_channels", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_audio", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """JSONL 构建 CLI 主流程。"""
    setup_logging()
    args = parse_args(argv)
    jsonl_root = Path(args.jsonl_root)
    build_split_jsonl(
        Path(args.train_index),
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
        Path(args.val_index),
        jsonl_root / "mvad_binary_val_with_audio.jsonl",
        Path(args.audio_root),
        args.ffmpeg,
        args.sample_rate,
        args.audio_channels,
        args.overwrite,
        args.dry_run,
        args.skip_audio,
    )
    logging.info("Wrote MVAD Qwen JSONL files to %s", jsonl_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
