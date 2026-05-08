"""
本文件功能：
- 解压 MVAD 原始目录中的 zip 文件，并生成解压 manifest。

主要内容：
- find_archives：递归发现 zip 文件。
- unpack_archives：解压 zip，记录每个压缩包状态。
- main：命令行入口。
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mvad.common import SUPPORTED_VIDEO_EXTENSIONS, write_json
from src.omniav_detect.evaluation.progress import create_progress


def setup_logging() -> None:
    """初始化日志格式。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def find_archives(source_root: Path) -> List[Path]:
    """递归查找 source_root 下的 zip 文件。"""
    return sorted(path for path in source_root.expanduser().rglob("*.zip") if path.is_file())


def archive_output_dir(archive_path: Path, source_root: Path, unpack_root: Path) -> Path:
    """根据 zip 相对路径生成稳定解压目录。"""
    relative = archive_path.resolve(strict=False).relative_to(source_root.resolve(strict=False))
    return unpack_root / relative.with_suffix("")


def count_extracted_videos(target_dir: Path) -> int:
    """统计解压目录内视频文件数量。"""
    return sum(1 for path in target_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS)


def extract_with_7z(archive_path: Path, target_dir: Path, overwrite: bool, extractor: str) -> None:
    """
    函数功能：
    - 调用 7z 解压 zip 文件。
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    overwrite_flag = "-aoa" if overwrite else "-aos"
    command = [
        extractor,
        "x",
        "-y",
        overwrite_flag,
        f"-o{target_dir}",
        str(archive_path),
    ]
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"7z failed for {archive_path} -> {target_dir}: {completed.stderr.strip() or completed.stdout.strip()}"
        )


def extract_with_zipfile(archive_path: Path, target_dir: Path) -> None:
    """使用 Python zipfile 解压，主要作为本地测试或无 7z 环境的后备方案。"""
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(target_dir)


def unpack_one_archive(
    archive_path: Path,
    source_root: Path,
    unpack_root: Path,
    overwrite: bool,
    extractor: str,
) -> Dict[str, Any]:
    """
    函数功能：
    - 解压单个 zip 文件并返回 manifest 记录。
    """
    target_dir = archive_output_dir(archive_path, source_root, unpack_root)
    done_marker = target_dir / ".unpacked"
    if done_marker.exists() and not overwrite:
        return {
            "archive_path": str(archive_path.resolve(strict=False)),
            "output_dir": str(target_dir.resolve(strict=False)),
            "status": "skipped",
            "video_count": count_extracted_videos(target_dir),
            "extractor": extractor,
        }
    if extractor == "zipfile":
        extract_with_zipfile(archive_path, target_dir)
    else:
        extract_with_7z(archive_path, target_dir, overwrite, extractor)
    done_marker.write_text("ok\n", encoding="utf-8")
    video_count = count_extracted_videos(target_dir)
    if video_count == 0:
        raise ValueError(f"No video files found after extracting {archive_path} to {target_dir}")
    return {
        "archive_path": str(archive_path.resolve(strict=False)),
        "output_dir": str(target_dir.resolve(strict=False)),
        "status": "extracted",
        "video_count": video_count,
        "extractor": extractor,
    }


def unpack_archives(
    source_root: Path,
    unpack_root: Path,
    overwrite: bool = False,
    extractor: str = "7z",
) -> List[Dict[str, Any]]:
    """
    函数功能：
    - 解压 source_root 下所有 zip 文件。

    参数：
    - source_root: MVAD 原始下载根目录。
    - unpack_root: 解压输出根目录。
    - overwrite: 是否覆盖已解压目录。

    返回：
    - 每个 zip 的 manifest 记录列表。
    """
    archives = find_archives(source_root)
    if not archives:
        raise ValueError(f"No zip archives found under {source_root}")
    manifest = []
    progress = create_progress(total=len(archives), desc="unpack mvad zip", unit="zip")
    try:
        for archive_path in archives:
            manifest.append(unpack_one_archive(archive_path, source_root, unpack_root, overwrite, extractor))
            progress.update(1)
    finally:
        progress.close()
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析解压命令行参数。"""
    parser = argparse.ArgumentParser(description="Unpack MVAD zip archives.")
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--unpack_root", required=True)
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--extractor", default="7z", help="Archive extractor executable, default 7z. Use zipfile for Python fallback.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """解压 CLI 主流程。"""
    setup_logging()
    args = parse_args(argv)
    manifest = unpack_archives(Path(args.source_root), Path(args.unpack_root), args.overwrite, args.extractor)
    manifest_path = Path(args.manifest_path) if args.manifest_path else Path(args.unpack_root) / "unpack_manifest.json"
    write_json(manifest_path, manifest)
    logging.info("Wrote unpack manifest to %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
