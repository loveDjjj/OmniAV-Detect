#!/usr/bin/env python3
"""
本文件功能：
- vLLM 后端的 Qwen2.5-Omni LoRA binary detector 单 checkpoint 评估入口。

主要内容：
- 将仓库 `src/` 加入导入路径。
- 调用 `omniav_detect.evaluation.binary_logits_vllm.main` 执行真实逻辑。
- 重新导出评估函数，兼容测试和旧导入方式。

使用方式：
- `python scripts/eval_binary_logits_qwen_omni_vllm.py --adapter_path ... --jsonl ... --output_dir ...`
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

from omniav_detect.evaluation.binary_logits_vllm import *  # noqa: F401,F403
from omniav_detect.evaluation.binary_logits_vllm import main


if __name__ == "__main__":
    raise SystemExit(main())
