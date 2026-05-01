"""
本文件功能：
- 提供项目内 YAML / JSON 配置文件读取的公共函数。

主要内容：
- load_config_file：按文件扩展名读取 `.yaml`、`.yml` 或 `.json` 配置。

使用方式：
- 数据准备入口和批量评估入口通过该模块加载配置文件。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config_file(path: Path | str) -> Dict[str, Any]:
    """
    函数功能：
    - 读取 YAML 或 JSON 配置文件，返回字典。

    参数：
    - path: 配置文件路径，支持 `.yaml`、`.yml` 和 `.json`。

    返回：
    - 配置字典。

    关键逻辑：
    - YAML 依赖 PyYAML；缺失时给出明确安装提示。
    - 顶层配置必须是 mapping，避免把列表或空文件误当成有效配置。
    """
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Reading YAML config requires PyYAML. Install it with: pip install pyyaml") from exc
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    elif suffix == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        raise ValueError(f"Unsupported config extension for {config_path}. Use .yaml, .yml, or .json.")

    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a top-level mapping/object")
    return payload
