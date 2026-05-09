"""
本文件功能：
- 提供 MVAD 预处理中的公共路径、标签和 JSONL 工具。

主要内容：
- parse_video_sample：从 MVAD 视频路径推断二分类标签、四类模态标签和 group_id。
- iter_video_files：递归扫描支持的视频文件。
- iter_audio_files：递归扫描支持的音频文件。
- is_ignored_extracted_path：过滤 macOS 解压元数据等非真实样本文件。
- write_json / write_jsonl：写出统计文件和 JSONL 文件。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
AUDIO_GENERATOR_TOKENS = ("audiox", "foleycrafter", "hunyuan", "mmaudio")
IGNORED_EXTRACTED_DIRS = {"__MACOSX"}

SYSTEM_PROMPT = "You are an audio-video deepfake detector. Given an input video, answer only Real or Fake."
BINARY_AUDIO_USER_PROMPT = (
    "<video>\n<audio>\nGiven the video and audio, please assess if it's Real or Fake? "
    "Only answer Real or Fake."
)

MODALITY_LABELS = {
    "real_real": ("Real", "Real", "Real", "R-R"),
    "real_fake": ("Fake", "Real", "Fake", "R-F"),
    "fake_real": ("Fake", "Fake", "Real", "F-R"),
    "fake_fake": ("Fake", "Fake", "Fake", "F-F"),
}


def abs_path(path: Path) -> str:
    """返回不要求真实存在的绝对路径字符串。"""
    return str(path.expanduser().resolve(strict=False))


def is_ignored_extracted_path(path: Path) -> bool:
    """
    函数功能：
    - 判断解压目录中的路径是否属于非样本文件。

    参数：
    - path: 待检查的视频或目录路径。

    返回：
    - true 表示该路径应从 MVAD 样本扫描中跳过。

    关键处理逻辑：
    - macOS zip 常带有 __MACOSX/._* AppleDouble 资源叉文件，扩展名可能也是 .mp4，
      但它们不是可解码视频，直接送入 ffmpeg 会出现 moov atom not found。
    """
    for part in path.parts:
        if part in IGNORED_EXTRACTED_DIRS:
            return True
        if part.startswith("._"):
            return True
    return False


def iter_video_files(root: Path) -> Iterable[Path]:
    """
    函数功能：
    - 递归扫描目录下所有支持的视频文件。

    参数：
    - root: 需要扫描的根目录。

    返回：
    - 视频文件路径迭代器。
    """
    root = root.expanduser()
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if is_ignored_extracted_path(path):
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            yield path


def iter_audio_files(root: Path) -> Iterable[Path]:
    """
    函数功能：
    - 递归扫描目录下所有支持的音频文件。
    """
    root = root.expanduser()
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if is_ignored_extracted_path(path):
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            yield path


def normalized_stem(path: Path) -> str:
    """把文件名主干归一化为适合做 group_id 的字符串。"""
    text = path.stem.lower()
    text = re.sub(r"[\s（）()]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def strip_audio_generator_suffix(stem: str) -> str:
    """
    函数功能：
    - 去掉文件名中表示音频生成器的后缀或片段。

    参数：
    - stem: 已归一化的文件主干。

    返回：
    - 去掉常见音频生成器标记后的 group 主体。
    """
    result = stem
    for token in AUDIO_GENERATOR_TOKENS:
        result = re.sub(rf"(^|_){re.escape(token)}($|_)", "_", result)
        result = re.sub(rf"_{re.escape(token)}$", "", result)
    result = re.sub(r"_+", "_", result).strip("_")
    return result or stem


def relative_parts(video_path: Path, root: Path) -> List[str]:
    """把视频路径转成相对于根目录的路径片段。"""
    try:
        return list(video_path.resolve(strict=False).relative_to(root.resolve(strict=False)).parts)
    except ValueError:
        return list(video_path.parts)


def infer_modality(parts: Sequence[str]) -> str:
    """从相对路径片段中查找 MVAD 四类模态目录。"""
    lowered = [part.lower() for part in parts]
    for modality in MODALITY_LABELS:
        if modality in lowered:
            return modality
    raise ValueError(f"Cannot infer MVAD modality from path parts: {parts}")


def infer_generation_path(modality: str, parts: Sequence[str]) -> str:
    """推断 fake_fake 样本的 direct/indirect 生成路径。"""
    lowered = [part.lower() for part in parts]
    if modality == "fake_fake":
        if "direct" in lowered:
            return "direct"
        if "indirect" in lowered:
            return "indirect"
    return ""


def infer_source_after(parts: Sequence[str], marker: str) -> str:
    """返回指定目录 marker 后面的第一个路径片段。"""
    lowered = [part.lower() for part in parts]
    if marker not in lowered:
        return ""
    idx = lowered.index(marker)
    if idx + 1 >= len(parts) - 1:
        return ""
    return parts[idx + 1]


def infer_audio_source(parts: Sequence[str]) -> str:
    """从目录名中推断音频生成器来源。"""
    for part in reversed(parts[:-1]):
        lowered = part.lower()
        for token in AUDIO_GENERATOR_TOKENS:
            if token in lowered:
                return part
    return ""


def build_group_id(modality: str, parts: Sequence[str], video_path: Path) -> str:
    """
    函数功能：
    - 为样本构造防泄漏划分使用的 group_id。

    参数：
    - modality: MVAD 四类模态目录名。
    - parts: 相对路径片段。
    - video_path: 视频绝对路径。

    返回：
    - 稳定的 group_id 字符串。
    """
    stem = normalized_stem(video_path)
    source = infer_source_after(parts, modality).lower() or "unknown"
    generation_path = infer_generation_path(modality, parts)
    if modality in {"real_fake"} or (modality == "fake_fake" and generation_path == "indirect"):
        stem = strip_audio_generator_suffix(stem)
    return f"{modality}:{source}:{generation_path}:{stem}"


def parse_video_sample(video_path: Path, root: Path) -> Dict[str, Any]:
    """
    函数功能：
    - 从 MVAD 视频路径推断训练所需标签和 metadata。

    参数：
    - video_path: 单个视频文件路径。
    - root: 解压后 MVAD 视频根目录。

    返回：
    - 包含 video_path 和 meta 的样本字典。
    """
    parts = relative_parts(video_path, root)
    modality = infer_modality(parts)
    overall_label, video_label, audio_label, compact_modality = MODALITY_LABELS[modality]
    meta = {
        "dataset": "MVAD",
        "source_path": abs_path(video_path),
        "relative_path": Path(*parts).as_posix(),
        "overall_label": overall_label,
        "video_label": video_label,
        "audio_label": audio_label,
        "modality_type": compact_modality,
        "mvad_modality": modality,
        "generation_path": infer_generation_path(modality, parts),
        "video_source": infer_source_after(parts, modality),
        "audio_source": infer_audio_source(parts),
    }
    meta["group_id"] = build_group_id(modality, parts, video_path)
    return {"video_path": abs_path(video_path), "meta": meta}


def make_binary_audio_record(sample: Dict[str, Any], audio_path: Path | str) -> Dict[str, Any]:
    """
    函数功能：
    - 生成 Qwen2.5-Omni / ms-swift 显式音频二分类 JSONL 记录。
    """
    meta = sample["meta"]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": BINARY_AUDIO_USER_PROMPT},
            {"role": "assistant", "content": meta["overall_label"]},
        ],
        "videos": [sample["video_path"]],
        "audios": [str(Path(audio_path).expanduser().resolve(strict=False))],
        "meta": meta,
    }


def output_audio_path(video_path: Path, audio_root: Path, suffix: str = ".wav") -> Path:
    """
    函数功能：
    - 为视频生成稳定音频输出路径。
    """
    drive = video_path.drive.rstrip(":").replace("\\", "/")
    relative_parts_for_path = [part for part in video_path.parts[1:-1] if part not in {"/", "\\"}]
    target_dir = audio_root / drive / Path(*relative_parts_for_path) if drive else audio_root / Path(*relative_parts_for_path)
    return target_dir / f"{video_path.stem}{suffix}"


def split_counts(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """统计样本标签和模态分布。"""
    labels = Counter(sample["meta"]["overall_label"] for sample in samples)
    modalities = Counter(sample["meta"]["modality_type"] for sample in samples)
    audio_handling = Counter(sample.get("audio_handling", "unknown") for sample in samples)
    groups = {sample["meta"]["group_id"] for sample in samples}
    return {
        "count": len(samples),
        "group_count": len(groups),
        "label_distribution": dict(sorted(labels.items())),
        "modality_type_distribution": dict(sorted(modalities.items())),
        "audio_handling_distribution": dict(sorted(audio_handling.items())),
    }


def write_json(path: Path, payload: Any) -> None:
    """写出 UTF-8 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """写出 UTF-8 JSONL 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
