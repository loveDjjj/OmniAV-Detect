"""
本文件功能：
- 为评估流程提供可选的 tqdm 进度条封装。

主要内容：
- count_batches：根据样本数和 batch size 计算批次数。
- ProgressProxy：在 tqdm 不可用时提供兼容的空实现。
- create_progress：优先创建 tqdm 进度条，失败时退化为空实现。

使用方式：
- 被单模型评估、批量评估和并行评估主流程调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def count_batches(num_samples: int, batch_size: int) -> int:
    """
    函数功能：
    - 计算给定样本数在指定 batch size 下的总批次数。
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if num_samples <= 0:
        return 0
    return (num_samples + batch_size - 1) // batch_size


@dataclass
class ProgressProxy:
    """
    函数功能：
    - 在 tqdm 不可用时提供兼容的进度条接口。
    """

    total: int | None = None
    desc: str | None = None
    unit: str | None = None
    n: int = 0

    def update(self, value: int = 1) -> None:
        self.n += value

    def close(self) -> None:
        return None


def create_progress(total: int | None, desc: str, unit: str) -> Any:
    """
    函数功能：
    - 创建 tqdm 进度条；如果环境缺少 tqdm，则退化为无输出代理对象。
    """
    try:
        from tqdm.auto import tqdm
    except Exception:
        return ProgressProxy(total=total, desc=desc, unit=unit)
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=True)
