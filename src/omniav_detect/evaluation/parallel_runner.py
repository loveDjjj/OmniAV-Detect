#!/usr/bin/env python3
"""
本文件功能：
- 负责 Qwen2.5-Omni binary logits 评估的外层多进程调度。

主要内容：
- split_jsonl_to_shards：把评估 JSONL 按 round-robin 切成多个 shard。
- build_worker_command / run_workers：按 GPU 启动多个单 checkpoint 评估子进程。
- write_merged_outputs：合并各 shard 的预测与失败样本，并重新计算指标。
- main：并行评估命令行入口。

使用方式：
- 通常通过 `python scripts/eval_parallel_binary_qwen_omni.py ...` 调用。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from omniav_detect.evaluation.data_io import write_json
from omniav_detect.evaluation.outputs import print_core_metrics, save_outputs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    函数功能：
    - 解析并行评估命令行参数。

    返回：
    - argparse.Namespace，包含模型、数据、GPU、worker 和输出参数。
    """
    parser = argparse.ArgumentParser(
        description="Parallel shard runner for Qwen2.5-Omni binary logits evaluation."
    )
    parser.add_argument("--model_path", default="/data/OneDay/models/qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--adapter_path", default=None, help="Optional LoRA checkpoint path. Omit for base-model evaluation.")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_script", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default=None, help="Comma separated GPU ids, e.g. 0,1. Defaults to CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of GPU worker processes.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--fake_token_id", type=int, default=52317)
    parser.add_argument("--real_token_id", type=int, default=12768)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--keep_shards", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--use_audio_in_video",
        dest="use_audio_in_video",
        action="store_true",
        help="Use the audio track embedded in each video. Enabled by default.",
    )
    parser.add_argument(
        "--no_use_audio_in_video",
        dest="use_audio_in_video",
        action="store_false",
        help="Disable audio extraction from video during evaluation.",
    )
    parser.set_defaults(use_audio_in_video=True)
    return parser.parse_args(argv)


def setup_logging() -> None:
    """初始化统一日志格式。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")


def default_eval_script() -> Path:
    """返回仓库内 Transformers 单模型评估脚本路径。"""
    return Path(__file__).resolve().parents[3] / "scripts" / "eval_binary_logits_qwen_omni.py"


def parse_gpu_list(gpus: Optional[str]) -> List[str]:
    """
    函数功能：
    - 解析命令行或环境变量中的 GPU id 列表。
    """
    raw = gpus if gpus is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw:
        items = [item.strip() for item in raw.split(",") if item.strip()]
        if items:
            return items
    return ["0"]


def resolve_worker_gpus(gpus: Optional[str], num_workers: Optional[int]) -> List[str]:
    """
    函数功能：
    - 根据 GPU 列表和 num_workers 计算实际 worker 使用的 GPU。

    关键逻辑：
    - 每个 worker 只暴露一张 GPU，通过 `CUDA_VISIBLE_DEVICES=<gpu>` 避免 `device_map=auto`
      把单个样本切到多卡上。
    """
    parsed = parse_gpu_list(gpus)
    if num_workers is None:
        return parsed
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    if num_workers > len(parsed):
        raise ValueError(f"num_workers={num_workers} exceeds available gpus {parsed}")
    return parsed[:num_workers]


def iter_jsonl_lines(jsonl_path: Path, max_samples: Optional[int]) -> Iterable[str]:
    """读取 JSONL 非空行，并按 max_samples 截断。"""
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_samples is not None and count >= max_samples:
                break
            if not line.strip():
                continue
            count += 1
            yield line if line.endswith("\n") else line + "\n"


def split_jsonl_to_shards(
    jsonl_path: Path | str,
    shard_dir: Path | str,
    num_shards: int,
    max_samples: Optional[int] = None,
) -> List[Path]:
    """
    函数功能：
    - 将输入 JSONL 切成多个 shard 文件。

    关键逻辑：
    - 使用 round-robin 分配样本，避免原始 JSONL 按类别排序时某个 worker 只拿到单一类别。
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    source = Path(jsonl_path)
    target_dir = Path(shard_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [target_dir / f"shard_{idx:03d}.jsonl" for idx in range(num_shards)]
    handles = [path.open("w", encoding="utf-8") for path in shard_paths]
    try:
        for idx, line in enumerate(iter_jsonl_lines(source, max_samples)):
            handles[idx % num_shards].write(line)
    finally:
        for handle in handles:
            handle.close()
    return shard_paths


def build_worker_command(
    python_executable: str,
    eval_script: Path,
    model_path: str,
    adapter_path: Optional[str],
    shard_jsonl: Path,
    shard_output_dir: Path,
    batch_size: int,
    fps: float,
    save_every: int,
    torch_dtype: str,
    device_map: str,
    fake_token_id: int,
    real_token_id: int,
    use_audio_in_video: bool,
    extra_args: Sequence[str],
) -> List[str]:
    """
    函数功能：
    - 构造单个 worker 的单模型评估命令。
    """
    command = [
        python_executable,
        str(eval_script),
        "--model_path",
        model_path,
        "--jsonl",
        str(shard_jsonl),
        "--output_dir",
        str(shard_output_dir),
        "--fake_token_id",
        str(fake_token_id),
        "--real_token_id",
        str(real_token_id),
        "--device_map",
        device_map,
        "--torch_dtype",
        torch_dtype,
        "--batch_size",
        str(batch_size),
        "--fps",
        str(fps),
        "--save_every",
        str(save_every),
    ]
    if adapter_path:
        command.extend(["--adapter_path", adapter_path])
    command.append("--use_audio_in_video" if use_audio_in_video else "--no_use_audio_in_video")
    command.extend(extra_args)
    return command


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件；不存在时返回空列表。"""
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collect_worker_outputs(worker_dirs: Sequence[Path]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    函数功能：
    - 汇总所有 worker 的 predictions.jsonl 和 bad_samples.jsonl。
    """
    predictions: List[Dict[str, Any]] = []
    bad_samples: List[Dict[str, Any]] = []
    for worker_index, worker_dir in enumerate(worker_dirs):
        for row in read_jsonl(worker_dir / "predictions.jsonl"):
            row.setdefault("parallel_worker", worker_index)
            predictions.append(row)
        for row in read_jsonl(worker_dir / "bad_samples.jsonl"):
            row.setdefault("parallel_worker", worker_index)
            bad_samples.append(row)
    return predictions, bad_samples


def write_merged_outputs(output_dir: Path | str, worker_dirs: Sequence[Path], run_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    函数功能：
    - 合并 worker 输出，并重新计算整体二分类指标。

    返回：
    - 合并后的 metrics 字典。
    """
    target = Path(output_dir)
    predictions, bad_samples = collect_worker_outputs(worker_dirs)
    return save_outputs(target, predictions, bad_samples, run_config)


def build_run_config(args: argparse.Namespace, worker_gpus: Sequence[str], worker_dirs: Sequence[Path]) -> Dict[str, Any]:
    """保存并行评估运行参数，写入最终 metrics.json。"""
    return {
        "backend": "transformers_parallel",
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "model_mode": "lora" if args.adapter_path else "base",
        "jsonl": args.jsonl,
        "max_samples": args.max_samples,
        "worker_gpus": list(worker_gpus),
        "num_workers": len(worker_gpus),
        "worker_dirs": [str(path) for path in worker_dirs],
        "fake_token_id": args.fake_token_id,
        "real_token_id": args.real_token_id,
        "device_map": args.device_map,
        "torch_dtype": args.torch_dtype,
        "batch_size": args.batch_size,
        "fps": args.fps,
        "use_audio_in_video": args.use_audio_in_video,
        "max_new_tokens": args.max_new_tokens,
        "save_every": args.save_every,
    }


def run_workers(
    commands: Sequence[List[str]],
    worker_gpus: Sequence[str],
    output_dir: Path,
    dry_run: bool,
) -> List[int]:
    """
    函数功能：
    - 并发启动所有 worker 子进程，并等待结束。
    """
    processes: List[tuple[int, subprocess.Popen[Any], Any]] = []
    for idx, (command, gpu) in enumerate(zip(commands, worker_gpus)):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_path = output_dir / f"worker_{idx:03d}.log"
        logging.info("Worker %d on GPU %s: %s", idx, gpu, shlex.join(command))
        if dry_run:
            continue
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
        processes.append((idx, process, log_handle))

    if dry_run:
        return [0 for _ in commands]

    returncodes = []
    for idx, process, log_handle in processes:
        try:
            returncode = process.wait()
            logging.info("Worker %d finished with returncode=%d", idx, returncode)
            returncodes.append(returncode)
        finally:
            log_handle.close()
    return returncodes


def write_parallel_manifest(output_dir: Path, commands: Sequence[List[str]], worker_gpus: Sequence[str]) -> None:
    """写出 worker 命令清单，便于复现与排查。"""
    rows = [
        {"worker": idx, "gpu": gpu, "command": command}
        for idx, (gpu, command) in enumerate(zip(worker_gpus, commands))
    ]
    write_json(output_dir / "parallel_manifest.json", {"workers": rows})


def cleanup_shards(shard_dir: Path, output_dir: Path) -> None:
    """
    函数功能：
    - 安全删除并行评估生成的临时 JSONL shard 目录。

    关键逻辑：
    - 只允许删除 output_dir 内的 shard_dir，避免误删用户数据。
    """
    if not shard_dir.exists():
        return
    resolved_shard = shard_dir.resolve()
    resolved_output = output_dir.resolve()
    if resolved_shard == resolved_output or resolved_output not in resolved_shard.parents:
        raise ValueError(f"Refusing to remove shard dir outside output_dir: {resolved_shard}")
    shutil.rmtree(resolved_shard)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    函数功能：
    - 并行评估 CLI 主流程。
    """
    args = parse_args(argv)
    setup_logging()
    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "shards"
    worker_root = output_dir / "workers"
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_root.mkdir(parents=True, exist_ok=True)

    worker_gpus = resolve_worker_gpus(args.gpus, args.num_workers)
    eval_script = Path(args.eval_script) if args.eval_script else default_eval_script()
    shard_paths = split_jsonl_to_shards(args.jsonl, shard_dir, len(worker_gpus), args.max_samples)
    worker_dirs = [worker_root / f"worker_{idx:03d}" for idx in range(len(worker_gpus))]
    for worker_dir in worker_dirs:
        worker_dir.mkdir(parents=True, exist_ok=True)

    extra_args = ["--max_new_tokens", str(args.max_new_tokens)]
    commands = [
        build_worker_command(
            python_executable=args.python,
            eval_script=eval_script,
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            shard_jsonl=shard_path,
            shard_output_dir=worker_dir,
            batch_size=args.batch_size,
            fps=args.fps,
            save_every=args.save_every,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            fake_token_id=args.fake_token_id,
            real_token_id=args.real_token_id,
            use_audio_in_video=args.use_audio_in_video,
            extra_args=extra_args,
        )
        for shard_path, worker_dir in zip(shard_paths, worker_dirs)
    ]
    write_parallel_manifest(output_dir, commands, worker_gpus)
    returncodes = run_workers(commands, worker_gpus, output_dir, args.dry_run)
    if args.dry_run:
        logging.info("Dry run enabled: shards and command manifest were written, workers were not started.")
        return 0
    if any(code != 0 for code in returncodes):
        write_json(output_dir / "parallel_status.json", {"returncodes": returncodes, "status": "failed"})
        return 1

    run_config = build_run_config(args, worker_gpus, worker_dirs)
    metrics = write_merged_outputs(output_dir, worker_dirs, run_config)
    metrics["parallel_returncodes"] = returncodes
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "parallel_status.json", {"returncodes": returncodes, "status": "completed"})
    if not args.keep_shards:
        cleanup_shards(shard_dir, output_dir)
    print_core_metrics(metrics)
    logging.info("Wrote merged predictions, bad samples, metrics, and visualizations to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
