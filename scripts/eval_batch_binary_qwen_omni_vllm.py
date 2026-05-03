#!/usr/bin/env python3
"""
本文件功能：
- vLLM 后端的 Qwen2.5-Omni 多 checkpoint 批量评估入口。

主要内容：
- 将仓库 `src/` 加入导入路径。
- 默认加载 `configs/eval/qwen_omni_binary_batch_eval_vllm.yaml`。
- 调用 `omniav_detect.evaluation.batch_runner.main` 执行真实逻辑。

使用方式：
- `python scripts/eval_batch_binary_qwen_omni_vllm.py --config configs/eval/qwen_omni_binary_batch_eval_vllm.yaml`
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    """将仓库 src 目录加入 sys.path，保证直接运行脚本时可导入项目包。"""
    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args(argv: list[str] | None = None):
    """复用 batch_runner 参数解析，但默认切换到 vLLM YAML 配置。"""
    _ensure_src_on_path()
    from omniav_detect.evaluation.batch_runner import parse_args as batch_parse_args

    argv = list(sys.argv[1:] if argv is None else argv)
    if "--config" not in argv:
        argv = ["--config", "configs/eval/qwen_omni_binary_batch_eval_vllm.yaml", *argv]
    return batch_parse_args(argv)


_ensure_src_on_path()

from omniav_detect.evaluation import batch_runner as _batch_runner


def main(argv: list[str] | None = None) -> int:
    """默认以 vLLM YAML 配置调用批量评估主流程。"""
    args = parse_args(argv)
    forwarded = [
        "--config",
        str(args.config),
        "--python",
        str(args.python),
    ]
    if args.output_root is not None:
        forwarded.extend(["--output_root", str(args.output_root)])
    if args.batch_size is not None:
        forwarded.extend(["--batch_size", str(args.batch_size)])
    if args.max_samples is not None:
        forwarded.extend(["--max_samples", str(args.max_samples)])
    if args.fps is not None:
        forwarded.extend(["--fps", str(args.fps)])
    if args.save_every is not None:
        forwarded.extend(["--save_every", str(args.save_every)])
    if args.dry_run:
        forwarded.append("--dry_run")
    if args.stop_on_error:
        forwarded.append("--stop_on_error")
    if args.only:
        forwarded.extend(["--only", *args.only])
    return _batch_runner.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
