#!/usr/bin/env python3
"""
本文件功能：
- ms-swift / Qwen2.5-Omni SFT 数据准备统一命令行入口。

主要内容：
- 将仓库 `src/` 加入导入路径。
- 调用 `omniav_detect.data.prepare_runner.main`，按 YAML 配置选择单个数据集。
- 重新导出公共函数和数据集 builder，兼容测试与交互式调试。

使用方式：
- `python scripts/prepare_swift_av_sft.py --dataset fakeavceleb --config configs/data/swift_av_sft.yaml`
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    """将仓库 src 目录加入 sys.path，保证兼容入口能导入项目包。"""
    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_ensure_src_on_path()

from omniav_detect.data.common import *  # noqa: F401,F403
from omniav_detect.data.fakeavceleb import (  # noqa: F401
    FAKEAVCELEB_CATEGORIES,
    build_fakeavceleb_fold_output_records,
    build_fakeavceleb_output_records,
    build_fakeavceleb_samples,
    stratified_split,
)
from omniav_detect.data import mavosdd  # noqa: F401
from omniav_detect.data.mavosdd import (  # noqa: F401
    MAVOS_OUTPUT_SPLITS,
    build_mavosdd_output_records,
    build_mavosdd_samples,
)
from omniav_detect.data.prepare_runner import *  # noqa: F401,F403
from omniav_detect.data.prepare_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
