"""
本文件功能：
- 负责 MVAD 解压目录中的音视频样本配对。

主要内容：
- build_audio_lookup：按目录和归一化文件名索引音频文件。
- video_has_audio_stream：用 ffprobe 判断视频容器是否有内嵌音轨。
- find_paired_audio：为单个视频寻找同目录同名音频文件。
- attach_audio_pairs：为样本补充 audio_path / audio_handling，并输出缺失报告。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from mvad.common import abs_path, iter_audio_files, normalized_stem


def path_key(path: Path) -> str:
    """把文件路径归一化为用于音视频配对的 key。"""
    return normalized_stem(path)


def build_audio_lookup(unpack_root: Path) -> Dict[Tuple[str, str], List[Path]]:
    """
    函数功能：
    - 扫描解压根目录，按父目录和归一化文件名建立音频索引。

    参数：
    - unpack_root: MVAD 解压根目录。

    返回：
    - key 为 (音频父目录绝对路径, 文件名 key)，value 为候选音频路径列表。
    """
    lookup: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for audio_path in iter_audio_files(unpack_root):
        parent = abs_path(audio_path.parent)
        lookup[(parent, path_key(audio_path))].append(audio_path)
    return lookup


def video_has_audio_stream(ffprobe: str, video_path: Path) -> bool:
    """
    函数功能：
    - 使用 ffprobe 判断视频文件中是否存在音频流。

    参数：
    - ffprobe: ffprobe 可执行程序。
    - video_path: 视频路径。

    返回：
    - true 表示视频容器里存在至少一条音频流。
    """
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return completed.returncode == 0 and "audio" in completed.stdout.lower()


def find_paired_audio(video_path: Path, audio_lookup: Dict[Tuple[str, str], List[Path]]) -> Optional[Path]:
    """
    函数功能：
    - 为单个视频寻找分离存放的音频文件。

    参数：
    - video_path: 视频路径。
    - audio_lookup: `build_audio_lookup` 生成的音频索引。

    返回：
    - 匹配到的音频路径；找不到则返回 None。
    """
    key = path_key(video_path)
    matches = audio_lookup.get((abs_path(video_path.parent), key), [])
    if matches:
        return sorted(matches)[0]
    return None


def build_missing_audio_row(sample: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """构造缺失音频报告行。"""
    meta = sample["meta"]
    return {
        "video_path": sample["video_path"],
        "relative_path": meta.get("relative_path", ""),
        "reason": reason,
        "overall_label": meta.get("overall_label", ""),
        "modality_type": meta.get("modality_type", ""),
        "mvad_modality": meta.get("mvad_modality", ""),
        "generation_path": meta.get("generation_path", ""),
    }


def attach_audio_pairs(
    samples: Sequence[Dict[str, Any]],
    unpack_root: Path,
    require_audio_pair: bool,
    ffprobe: str = "ffprobe",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    函数功能：
    - 为 MVAD 样本补充分离音频路径，必要时过滤缺失音频样本。

    参数：
    - samples: 已解析出视频和标签的样本。
    - unpack_root: MVAD 解压根目录。
    - require_audio_pair: true 时找不到音频配对的样本会进入缺失报告并被过滤。
    - ffprobe: 用于检测视频内嵌音轨的 ffprobe 可执行程序。

    返回：
    - 可用样本列表和缺失音频报告列表。
    """
    audio_lookup = build_audio_lookup(unpack_root)
    paired_samples: List[Dict[str, Any]] = []
    missing_audio: List[Dict[str, Any]] = []
    for sample in samples:
        video_path = Path(sample["video_path"]).expanduser().resolve(strict=False)
        paired_audio = find_paired_audio(video_path, audio_lookup)
        if paired_audio:
            enriched = dict(sample)
            enriched["audio_path"] = abs_path(paired_audio)
            enriched["audio_handling"] = "paired_file"
            paired_samples.append(enriched)
            continue
        if video_has_audio_stream(ffprobe, video_path):
            enriched = dict(sample)
            enriched["audio_path"] = abs_path(video_path.with_suffix(".wav"))
            enriched["audio_handling"] = "extract_from_video"
            paired_samples.append(enriched)
            continue
        if require_audio_pair:
            missing_audio.append(build_missing_audio_row(sample, "missing_audio"))
            continue
        enriched = dict(sample)
        enriched["audio_path"] = ""
        enriched["audio_handling"] = "extract_from_video"
        paired_samples.append(enriched)
    return paired_samples, missing_audio
