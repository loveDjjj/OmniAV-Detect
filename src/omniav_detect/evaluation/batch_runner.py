#!/usr/bin/env python3
"""
本文件功能：
- 根据 YAML / JSON 配置批量运行多个 Qwen2.5-Omni binary logits 评估任务。

主要内容：
- load_config / resolve_run：读取并展开批量评估配置。
- build_eval_command：生成单 checkpoint 评估命令。
- write_summary：汇总每个 run 的 metrics.json 到 CSV/JSON。
- main：批量评估命令行入口。

使用方式：
- 通常通过 `python scripts/eval_batch_binary_qwen_omni.py --config ...` 调用。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import posixpath
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from omniav_detect.config import load_config_file


SUMMARY_FIELDS = [
    "name",
    "dataset",
    "status",
    "returncode",
    "accuracy",
    "auc",
    "ap",
    "map",
    "fake_recall",
    "real_recall",
    "num_predictions",
    "num_bad_samples",
    "confusion_matrix_labels",
    "confusion_matrix",
    "adapter_path",
    "jsonl",
    "output_dir",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch runner for Qwen2.5-Omni binary logits evaluation.")
    parser.add_argument("--config", default="configs/eval/qwen_omni_binary_batch_eval.yaml")
    parser.add_argument("--only", nargs="*", default=None, help="Run only the named config entries.")
    parser.add_argument("--output_root", default=None, help="Override config output_root.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override all run batch sizes.")
    parser.add_argument("--max_samples", type=int, default=None, help="Override all run max_samples.")
    parser.add_argument("--fps", type=float, default=None, help="Override all run fps values.")
    parser.add_argument("--save_every", type=int, default=None, help="Override all run save_every values.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--eval_script", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    return parser.parse_args(argv)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path: Path | str) -> Dict[str, Any]:
    """
    函数功能：
    - 读取批量评估 YAML 或 JSON 配置。

    参数：
    - path: YAML 或 JSON 配置文件路径。

    返回：
    - 配置字典。

    关键逻辑：
    - 要求配置中存在非空 runs 列表，避免 dry-run 时静默无任务。
    """
    config_path = Path(path)
    config = load_config_file(config_path)
    if not isinstance(config.get("runs"), list) or not config["runs"]:
        raise ValueError(f"{config_path} must contain a non-empty 'runs' list")
    return config


def normalize_only(values: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not values:
        return None
    names: set[str] = set()
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                names.add(item)
    return names or None


def resolve_run(config: Dict[str, Any], run: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数功能：
    - 合并全局默认值、单个 run 配置和命令行覆盖参数。

    参数：
    - config: 批量配置整体字典。
    - run: 单个评估任务配置。
    - overrides: 命令行传入的覆盖参数。

    返回：
    - 可直接传给单模型评估脚本的完整 run 配置。

    关键逻辑：
    - 不修改配置文件本身，只在运行时计算 output_dir、batch_size、fps 等最终值。
    """
    resolved: Dict[str, Any] = {}
    resolved.update(config.get("defaults", {}))
    resolved.update(run)
    if "model_path" not in resolved and "model_path" in config:
        resolved["model_path"] = config["model_path"]

    output_root = overrides.get("output_root") or resolved.get("output_root") or config.get("output_root")
    if not output_root:
        output_root = "outputs/batch_eval"
    if overrides.get("batch_size") is not None:
        resolved["batch_size"] = overrides["batch_size"]
    if overrides.get("max_samples") is not None:
        resolved["max_samples"] = overrides["max_samples"]
    if overrides.get("fps") is not None:
        resolved["fps"] = overrides["fps"]
    if overrides.get("save_every") is not None:
        resolved["save_every"] = overrides["save_every"]

    resolved.setdefault("batch_size", 1)
    resolved.setdefault("fps", 1.0)
    resolved.setdefault("torch_dtype", "bfloat16")
    resolved.setdefault("device_map", "auto")
    resolved.setdefault("use_audio_in_video", True)
    resolved.setdefault("save_every", 100)
    resolved.setdefault("fake_token_id", 52317)
    resolved.setdefault("real_token_id", 12768)

    if not resolved.get("output_dir"):
        resolved["output_dir"] = join_output_dir(str(output_root), str(resolved.get("name", "unnamed_run")))

    missing = [key for key in ["name", "model_path", "jsonl", "output_dir"] if not resolved.get(key)]
    if missing:
        raise ValueError(f"Run config is missing required fields {missing}: {run}")
    return resolved


def join_output_dir(output_root: str, run_name: str) -> str:
    if "/" in output_root and "\\" not in output_root:
        return posixpath.join(output_root, run_name)
    return str(Path(output_root) / run_name)


def build_eval_command(run: Dict[str, Any], python_executable: str, eval_script: Path) -> List[str]:
    """
    函数功能：
    - 将单个 run 配置转换为可执行的单 checkpoint 评估命令。

    参数：
    - run: resolve_run 输出的完整 run 配置。
    - python_executable: Python 解释器路径。
    - eval_script: 单模型评估脚本路径。

    返回：
    - subprocess.run 可直接使用的命令参数列表。

    关键逻辑：
    - 显式传递 batch_size、fps、token id 和音频开关，保证批量运行可复现。
    """
    command = [
        python_executable,
        str(eval_script),
        "--model_path",
        str(run["model_path"]),
        "--jsonl",
        str(run["jsonl"]),
        "--output_dir",
        str(run["output_dir"]),
        "--fake_token_id",
        str(run.get("fake_token_id", 52317)),
        "--real_token_id",
        str(run.get("real_token_id", 12768)),
        "--device_map",
        str(run.get("device_map", "auto")),
        "--torch_dtype",
        str(run.get("torch_dtype", "bfloat16")),
        "--batch_size",
        str(run.get("batch_size", 1)),
        "--fps",
        str(run.get("fps", 1.0)),
        "--save_every",
        str(run.get("save_every", 100)),
    ]
    if run.get("adapter_path"):
        command.extend(["--adapter_path", str(run["adapter_path"])])
    if run.get("max_samples") is not None:
        command.extend(["--max_samples", str(run["max_samples"])])
    command.append("--use_audio_in_video" if run.get("use_audio_in_video", True) else "--no_use_audio_in_video")
    return command


def read_metrics(output_dir: Path | str) -> Dict[str, Any]:
    metrics_path = Path(output_dir) / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def build_summary_row(run: Dict[str, Any], returncode: Optional[int], status: str) -> Dict[str, Any]:
    metrics = read_metrics(run["output_dir"])
    return {
        "name": run.get("name"),
        "dataset": run.get("dataset", ""),
        "status": status,
        "returncode": returncode,
        "accuracy": metrics.get("accuracy"),
        "auc": metrics.get("roc_auc"),
        "ap": metrics.get("average_precision"),
        "map": metrics.get("map"),
        "fake_recall": metrics.get("fake_recall"),
        "real_recall": metrics.get("real_recall"),
        "num_predictions": metrics.get("num_predictions"),
        "num_bad_samples": metrics.get("num_bad_samples"),
        "confusion_matrix_labels": metrics.get("confusion_matrix_labels"),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "adapter_path": run.get("adapter_path"),
        "jsonl": run.get("jsonl"),
        "output_dir": run.get("output_dir"),
    }


def write_summary(output_root: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """
    函数功能：
    - 写出批量评估汇总 JSON 和 CSV。

    参数：
    - output_root: 批量评估输出根目录。
    - rows: 每个 run 的汇总行。

    返回：
    - 无返回值，直接写 `batch_eval_summary.json/csv`。

    关键逻辑：
    - CSV 中的复杂字段以 JSON 字符串保存，方便表格软件查看。
    """
    output_root.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    (output_root / "batch_eval_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_root / "batch_eval_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            for key in ["confusion_matrix_labels", "confusion_matrix"]:
                csv_row[key] = json.dumps(csv_row.get(key), ensure_ascii=False)
            writer.writerow(csv_row)


def default_eval_script() -> Path:
    """
    函数功能：
    - 返回仓库内单 checkpoint 评估脚本的默认路径。

    参数：
    - 无。

    返回：
    - `scripts/eval_binary_logits_qwen_omni.py` 的 Path。

    关键逻辑：
    - batch_runner 位于 `src/omniav_detect/evaluation/`，因此向上三级回到仓库根目录。
    """
    return Path(__file__).resolve().parents[3] / "scripts" / "eval_binary_logits_qwen_omni.py"


def iter_resolved_runs(config: Dict[str, Any], only: Optional[set[str]], overrides: Dict[str, Any]) -> List[Dict[str, Any]]:
    resolved_runs = []
    for run in config["runs"]:
        if only is not None and run.get("name") not in only:
            continue
        resolved_runs.append(resolve_run(config, run, overrides))
    if not resolved_runs:
        raise ValueError("No runs selected. Check --only or the config runs list.")
    return resolved_runs


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    函数功能：
    - 批量评估 CLI 主流程。

    参数：
    - argv: 命令行参数列表，None 时读取真实命令行。

    返回：
    - 进程退出码，全部完成或 dry-run 为 0，否则为 1 或失败命令退出码。

    关键逻辑：
    - 每个 checkpoint 使用独立子进程评估，避免连续加载多个 LoRA 时显存状态互相影响。
    """
    args = parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    overrides = {
        "output_root": args.output_root,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "fps": args.fps,
        "save_every": args.save_every,
    }
    only = normalize_only(args.only)
    runs = iter_resolved_runs(config, only, overrides)
    output_root = Path(args.output_root or config.get("output_root", "outputs/batch_eval"))
    eval_script = Path(args.eval_script) if args.eval_script else default_eval_script()
    summary_rows: List[Dict[str, Any]] = []

    logging.info("Loaded %d batch eval runs from %s", len(runs), args.config)
    for run in runs:
        command = build_eval_command(run, python_executable=args.python, eval_script=eval_script)
        logging.info("Run %s: %s", run["name"], shlex.join(command))
        if args.dry_run:
            summary_rows.append(build_summary_row(run, returncode=None, status="dry_run"))
            continue

        completed = subprocess.run(command, check=False)
        status = "completed" if completed.returncode == 0 else "failed"
        summary_rows.append(build_summary_row(run, returncode=completed.returncode, status=status))
        write_summary(output_root, summary_rows)
        if completed.returncode != 0 and args.stop_on_error:
            logging.error("Stopping after failed run %s", run["name"])
            return completed.returncode

    if args.dry_run:
        logging.info("Dry run enabled: commands were printed and no summary files were written.")
        return 0

    write_summary(output_root, summary_rows)
    logging.info("Wrote batch summaries to %s", output_root)
    return 0 if all(row["status"] in {"completed", "dry_run"} for row in summary_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
