#!/usr/bin/env python3
"""
本文件功能：
- Qwen2.5-Omni binary detector 并行评估入口。

主要内容：
- 将仓库 `src/` 加入导入路径。
- 调用 `omniav_detect.evaluation.parallel_runner.main` 执行 JSONL 分片、多 GPU worker 调度和结果合并。

使用方式：
- `python scripts/eval_parallel_binary_qwen_omni.py --jsonl ... --output_dir ... --gpus 0,1`
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    """将仓库 src 目录加入 sys.path，保证直接运行脚本时可导入项目包。"""
    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_ensure_src_on_path()

from omniav_detect.evaluation.parallel_runner import *  # noqa: F401,F403
from omniav_detect.evaluation.parallel_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
