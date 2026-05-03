"""
本文件功能：
- 根据 YAML / JSON 配置批量运行 Qwen2.5-Omni binary logits 评估任务。

主要内容：
- load_config / resolve_run：读取并展开批量评估配置。
- build_eval_command：按并行或 vLLM 后端生成评估命令。
- write_summary：汇总每个 run 的 `metrics.json` 到 CSV/JSON。
- main：批量评估命令行主流程。

使用方式：
- 通常通过 `python scripts/eval_batch_binary_qwen_omni.py --config ...`
  或 `python scripts/eval_batch_binary_qwen_omni_vllm.py --config ...` 调用。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import posixpath
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from omniav_detect.config import load_config_file
from omniav_detect.evaluation.progress import create_progress


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
    """
    函数功能：
    - 解析批量评估命令行参数。

    参数：
    - argv: 可选命令行参数列表；为 None 时读取真实命令行。

    返回：
    - argparse.Namespace，包含配置路径、批量覆盖参数和 dry-run 控制项。
    """
    parser = argparse.ArgumentParser(description="Batch runner for Qwen2.5-Omni binary logits evaluation.")
    parser.add_argument("--config", default="configs/eval/qwen_omni_binary_batch_eval.yaml")
    parser.add_argument("--only", nargs="*", default=None, help="Run only the named config entries.")
    parser.add_argument("--output_root", default=None, help="Override config output_root.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override all run batch sizes.")
    parser.add_argument("--max_samples", type=int, default=None, help="Override all run max_samples.")
    parser.add_argument("--fps", type=float, default=None, help="Override all run fps values.")
    parser.add_argument("--save_every", type=int, default=None, help="Override all run save_every values.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    return parser.parse_args(argv)


def setup_logging() -> None:
    """初始化统一日志格式。"""
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
    - path: 配置文件路径。

    返回：
    - 配置字典。

    关键逻辑：
    - 要求配置中存在非空 `runs` 列表，避免 dry-run 时静默无任务。
    """
    config_path = Path(path)
    config = load_config_file(config_path)
    if not isinstance(config.get("runs"), list) or not config["runs"]:
        raise ValueError(f"{config_path} must contain a non-empty 'runs' list")
    return config


def normalize_only(values: Optional[Sequence[str]]) -> Optional[set[str]]:
    """
    函数功能：
    - 规范化 `--only` 传入的 run 名称列表。

    参数：
    - values: 命令行传入的字符串列表。

    返回：
    - 去重后的名称集合；为空时返回 None。
    """
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
    - 可直接传给评估后端的完整 run 配置。

    关键逻辑：
    - 不修改配置文件本身，只在运行时计算 `output_dir`、`batch_size`、`fps` 等最终值。
    """
    resolved: Dict[str, Any] = {}
    resolved.update(config.get("defaults", {}))
    resolved.update(run)
    resolved["eval_backend"] = str(
        resolved.get("eval_backend") or config.get("eval_backend") or "parallel"
    ).strip().lower()
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
    if resolved["eval_backend"] == "parallel":
        resolved.setdefault("gpus", "0,1")
        resolved.setdefault("num_workers", None)
    elif resolved["eval_backend"] == "vllm":
        resolved.setdefault("tensor_parallel_size", 1)
        resolved.setdefault("max_model_len", None)
        resolved.setdefault("gpu_memory_utilization", 0.9)
        resolved.setdefault("trust_remote_code", True)
        resolved.setdefault("mm_format", "omni_av")
        resolved.setdefault("temperature", 0.0)
        resolved.setdefault("top_p", 1.0)
        resolved.setdefault("logprobs", -1)
    else:
        raise ValueError(f"Unsupported eval_backend={resolved['eval_backend']!r}; expected 'parallel' or 'vllm'")

    if not resolved.get("output_dir"):
        resolved["output_dir"] = join_output_dir(str(output_root), str(resolved.get("name", "unnamed_run")))

    missing = [key for key in ["name", "model_path", "jsonl", "output_dir"] if not resolved.get(key)]
    if missing:
        raise ValueError(f"Run config is missing required fields {missing}: {run}")
    return resolved


def join_output_dir(output_root: str, run_name: str) -> str:
    """
    函数功能：
    - 生成单个 run 的输出目录路径。

    参数：
    - output_root: 批量输出根目录。
    - run_name: run 名称。

    返回：
    - 平台兼容的输出目录字符串。
    """
    if "/" in output_root and "\\" not in output_root:
        return posixpath.join(output_root, run_name)
    return str(Path(output_root) / run_name)


def build_eval_command(run: Dict[str, Any], python_executable: str, eval_module: str) -> List[str]:
    """
    函数功能：
    - 将单个 run 配置转换为可执行的评估命令。

    参数：
    - run: `resolve_run` 输出的完整 run 配置。
    - python_executable: Python 解释器路径。
    - eval_module: 评估后端对应的内部模块入口。

    返回：
    - `subprocess.run` 可直接使用的命令参数列表。

    关键逻辑：
    - 统一走 `python -m ...` 内部模块入口，只保留两个对外 `scripts/` 入口。
    """
    backend = str(run.get("eval_backend", "parallel")).strip().lower()
    command = [
        python_executable,
        "-m",
        str(eval_module),
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

    if backend == "parallel":
        command.extend(
            [
                "--gpus",
                str(run.get("gpus", "0,1")),
                "--device_map",
                str(run.get("device_map", "auto")),
            ]
        )
        if run.get("num_workers") is not None:
            command.extend(["--num_workers", str(run["num_workers"])])
        return command

    if backend == "vllm":
        command.extend(
            [
                "--device_map",
                "vllm",
                "--tensor_parallel_size",
                str(run.get("tensor_parallel_size", 1)),
                "--gpu_memory_utilization",
                str(run.get("gpu_memory_utilization", 0.9)),
                "--mm_format",
                str(run.get("mm_format", "omni_av")),
                "--temperature",
                str(run.get("temperature", 0.0)),
                "--top_p",
                str(run.get("top_p", 1.0)),
                "--logprobs",
                str(run.get("logprobs", -1)),
            ]
        )
        if run.get("max_model_len") is not None:
            command.extend(["--max_model_len", str(run["max_model_len"])])
        command.append("--trust_remote_code" if run.get("trust_remote_code", True) else "--no_trust_remote_code")
        return command

    raise ValueError(f"Unsupported eval_backend={backend!r}")


def build_subprocess_env() -> Dict[str, str]:
    """
    函数功能：
    - 为批量评估子进程补齐 `PYTHONPATH`。

    返回：
    - 可直接传给 `subprocess.run` 的环境变量字典。

    关键逻辑：
    - 将仓库 `src/` 目录加入 `PYTHONPATH`，避免依赖 editable install。
    """
    env = os.environ.copy()
    repo_src = str(Path(__file__).resolve().parents[3] / "src")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_src if not current else f"{repo_src}{os.pathsep}{current}"
    return env


def read_metrics(output_dir: Path | str) -> Dict[str, Any]:
    """
    函数功能：
    - 读取单个 run 的 `metrics.json`。

    参数：
    - output_dir: 评估输出目录。

    返回：
    - 指标字典；不存在时返回空字典。
    """
    metrics_path = Path(output_dir) / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def build_summary_row(run: Dict[str, Any], returncode: Optional[int], status: str) -> Dict[str, Any]:
    """
    函数功能：
    - 为单个 run 构造批量汇总行。
    """
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


def default_eval_module(eval_backend: str = "parallel") -> str:
    """
    函数功能：
    - 返回与评估后端匹配的内部模块入口。

    参数：
    - eval_backend: 后端名称，支持 `parallel` 和 `vllm`。

    返回：
    - Python `-m` 调用所需的模块名字符串。
    """
    backend = str(eval_backend).strip().lower()
    if backend == "parallel":
        return "omniav_detect.evaluation.parallel_cli"
    if backend == "vllm":
        return "omniav_detect.evaluation.vllm_cli"
    raise ValueError(f"Unsupported eval_backend={eval_backend!r}")


def iter_resolved_runs(config: Dict[str, Any], only: Optional[set[str]], overrides: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    函数功能：
    - 按 `--only` 过滤并展开所有待执行 run。

    参数：
    - config: 批量配置整体字典。
    - only: 允许执行的 run 名称集合。
    - overrides: 命令行覆盖参数。

    返回：
    - 展开后的 run 列表。
    """
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
    - 批量评估命令行主流程。

    参数：
    - argv: 可选命令行参数列表；为 None 时读取真实命令行。

    返回：
    - 进程退出码；全部完成或 dry-run 时为 0，否则为 1 或失败命令的退出码。

    关键逻辑：
    - 每个 checkpoint 使用独立子进程评估，避免连续加载多个模型时状态互相影响。
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
    summary_rows: List[Dict[str, Any]] = []

    logging.info("Loaded %d batch eval runs from %s", len(runs), args.config)
    progress = create_progress(total=len(runs), desc="Batch eval runs", unit="run")
    env = build_subprocess_env()
    try:
        for run in runs:
            eval_module = default_eval_module(run.get("eval_backend", "parallel"))
            command = build_eval_command(run, python_executable=args.python, eval_module=eval_module)
            logging.info("Run %s: %s", run["name"], shlex.join(command))
            if args.dry_run:
                summary_rows.append(build_summary_row(run, returncode=None, status="dry_run"))
                progress.update(1)
                continue

            completed = subprocess.run(command, check=False, env=env)
            status = "completed" if completed.returncode == 0 else "failed"
            summary_rows.append(build_summary_row(run, returncode=completed.returncode, status=status))
            write_summary(output_root, summary_rows)
            progress.update(1)
            if completed.returncode != 0 and args.stop_on_error:
                logging.error("Stopping after failed run %s", run["name"])
                return completed.returncode
    finally:
        progress.close()

    if args.dry_run:
        logging.info("Dry run enabled: commands were printed and no summary files were written.")
        return 0

    write_summary(output_root, summary_rows)
    logging.info("Wrote batch summaries to %s", output_root)
    return 0 if all(row["status"] in {"completed", "dry_run"} for row in summary_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
