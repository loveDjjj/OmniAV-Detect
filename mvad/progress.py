"""
本文件功能：
- 为 MVAD 解压、扫描和音频处理流程提供可选的 tqdm 进度条封装。

主要内容：
- ProgressProxy：在 tqdm 不可用时提供兼容的空实现。
- create_progress：优先创建 tqdm 进度条，失败时退化为空实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProgressProxy:
    """
    函数功能：
    - 在 tqdm 不可用时提供兼容的进度条接口。

    参数：
    - total: 任务总数。
    - desc: 进度条描述。
    - unit: 单位名称。

    返回：
    - 无返回值；仅维护已更新数量。
    """

    total: int | None = None
    desc: str | None = None
    unit: str | None = None
    n: int = 0

    def update(self, value: int = 1) -> None:
        """更新已完成数量。"""
        self.n += value

    def close(self) -> None:
        """关闭空进度条。"""
        return None


def create_progress(total: int | None, desc: str, unit: str) -> Any:
    """
    函数功能：
    - 创建 tqdm 进度条；如果环境缺少 tqdm，则退化为无输出代理对象。

    参数：
    - total: 任务总数。
    - desc: 进度条描述。
    - unit: 单位名称。

    返回：
    - tqdm 实例或 ProgressProxy 实例。
    """
    try:
        from tqdm.auto import tqdm
    except Exception:
        return ProgressProxy(total=total, desc=desc, unit=unit)
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=True)
