"""
本文件功能：
- 提供 Transformers 单 worker 评估的内部命令行入口。

主要内容：
- main：转调 `binary_logits.main`，供并行评估 worker 子进程使用。

使用方式：
- 仅供内部通过 `python -m omniav_detect.evaluation.worker_cli ...` 调用。
"""

from __future__ import annotations

from typing import Optional, Sequence

from omniav_detect.evaluation.binary_logits import main as binary_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    函数功能：
    - 转调 Transformers 单 worker 评估主流程。

    参数：
    - argv: 可选命令行参数列表；为 None 时读取真实命令行。

    返回：
    - 评估进程退出码。

    关键逻辑：
    - 不实现额外业务逻辑，只作为内部稳定入口，避免对 `scripts/` 目录产生依赖。
    """
    return binary_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
