"""
本文件功能：
- 提供并行评估后端的内部命令行入口。

主要内容：
- main：转调 `parallel_runner.main`，供批量评估脚本按模块方式启动。

使用方式：
- 仅供内部通过 `python -m omniav_detect.evaluation.parallel_cli ...` 调用。
"""

from __future__ import annotations

from typing import Optional, Sequence

from omniav_detect.evaluation.parallel_runner import main as parallel_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    函数功能：
    - 转调并行评估主流程。

    参数：
    - argv: 可选命令行参数列表；为 None 时读取真实命令行。

    返回：
    - 评估进程退出码。

    关键逻辑：
    - 仅作为批量评估入口调用的内部模块入口，避免额外脚本层。
    """
    return parallel_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
