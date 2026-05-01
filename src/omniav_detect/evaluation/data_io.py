"""
本文件功能：
- 负责 Qwen2.5-Omni binary 评估所需的 JSONL 样本读取和基础文件写出。

主要内容：
- load_jsonl_samples：读取 ms-swift JSONL 并抽取视频路径与标签。
- batch_samples：按 batch_size 分组样本。
- write_jsonl / write_json：评估输出文件写入工具。

使用方式：
- 被 `binary_logits.py` 主流程和输出模块复用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from omniav_detect.evaluation.metrics import normalize_label


def normalize_video_path(raw_path: str, jsonl_path: Path) -> str:
    """将 JSONL 中的相对或绝对视频路径规范化为可读取路径。"""
    if raw_path.startswith("/"):
        return raw_path
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return str(path)
    return str((jsonl_path.parent / path).resolve())


def load_jsonl_samples(jsonl_path: Path | str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    函数功能：
    - 读取 ms-swift JSONL 测试集并抽取视频路径与 Real/Fake 标签。

    参数：
    - jsonl_path: 测试集 JSONL 路径。
    - max_samples: 最多读取的样本数，None 表示读取全部。

    返回：
    - 评估样本列表，包含 index、line_number、video_path、label、meta 和原始记录。

    关键逻辑：
    - 以 `meta.overall_label` 作为真值标签，缺少视频或标签非法时直接报错。
    """
    path = Path(jsonl_path)
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_samples is not None and len(samples) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            videos = record.get("videos") or []
            meta = record.get("meta") or {}
            label = normalize_label(meta.get("overall_label"))
            if not videos:
                raise ValueError(f"{path}:{line_number} has no videos field")
            if label not in {"Real", "Fake"}:
                raise ValueError(f"{path}:{line_number} has invalid meta.overall_label={meta.get('overall_label')!r}")
            samples.append(
                {
                    "index": len(samples),
                    "line_number": line_number,
                    "video_path": normalize_video_path(str(videos[0]), path),
                    "label": label,
                    "meta": meta,
                    "source_record": record,
                }
            )
    return samples


def batch_samples(samples: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    """
    函数功能：
    - 将样本按 batch_size 分组，末尾不足一个 batch 时保留尾部。
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    for start in range(0, len(samples), batch_size):
        yield list(samples[start : start + batch_size])


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """逐行写 JSONL 文件。"""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """写缩进 JSON 文件，便于人工检查。"""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
