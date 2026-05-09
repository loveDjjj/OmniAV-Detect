"""
本文件功能：
- 负责 MVAD 解压目录中的音视频样本配对。

主要内容：
- build_audio_lookup：按目录和归一化文件名索引音频文件。
- find_paired_audio：为单个视频寻找同级结构中的音频文件。
- attach_audio_pairs：为样本补充 audio_path / audio_handling，并输出缺失报告。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from mvad.common import abs_path, iter_audio_files, normalized_stem


VIDEO_DIR_NAMES = {"video", "videos"}
AUDIO_DIR_NAME_BY_VIDEO_DIR = {"video": "audio", "videos": "audios"}


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


def candidate_audio_dirs(video_path: Path) -> List[Path]:
    """
    函数功能：
    - 基于 MVAD 常见 `videos/audios` 或 `video/audio` 目录结构推断候选音频目录。
    """
    candidates: List[Path] = []
    for idx, part in enumerate(video_path.parts):
        lowered = part.lower()
        if lowered not in VIDEO_DIR_NAMES:
            continue
        audio_dir_name = AUDIO_DIR_NAME_BY_VIDEO_DIR[lowered]
        candidate = Path(*video_path.parts[:idx], audio_dir_name, *video_path.parts[idx + 1 : -1])
        candidates.append(candidate)
    return candidates


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
    for audio_dir in candidate_audio_dirs(video_path):
        matches = audio_lookup.get((abs_path(audio_dir), key), [])
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
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    函数功能：
    - 为 MVAD 样本补充分离音频路径，必要时过滤缺失音频样本。

    参数：
    - samples: 已解析出视频和标签的样本。
    - unpack_root: MVAD 解压根目录。
    - require_audio_pair: true 时找不到音频配对的样本会进入缺失报告并被过滤。

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
        if require_audio_pair:
            missing_audio.append(build_missing_audio_row(sample, "missing_audio_pair"))
            continue
        enriched = dict(sample)
        enriched["audio_path"] = ""
        enriched["audio_handling"] = "extract_from_video"
        paired_samples.append(enriched)
    return paired_samples, missing_audio
