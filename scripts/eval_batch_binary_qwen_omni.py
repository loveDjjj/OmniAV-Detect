#!/usr/bin/env python3
"""
本文件功能：
- Qwen2.5-Omni binary detector 多 checkpoint 批量评估入口。

主要内容：
- 将仓库 `src/` 加入导入路径。
- 调用 `omniav_detect.evaluation.batch_runner.main` 执行真实逻辑。
- 重新导出批量评估函数，兼容已有测试和旧导入方式。

使用方式：
- `python scripts/eval_batch_binary_qwen_omni.py --config configs/eval/qwen_omni_binary_batch_eval.yaml`
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

from omniav_detect.evaluation.batch_runner import *  # noqa: F401,F403
from omniav_detect.evaluation.batch_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
