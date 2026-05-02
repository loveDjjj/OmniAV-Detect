#!/usr/bin/env python3
"""
本文件功能：
- 组织 vLLM 后端的 Qwen2.5-Omni LoRA binary deepfake detector 单 checkpoint 评估主流程。

主要内容：
- parse_args / setup_logging：命令行参数和日志初始化。
- main：读取 JSONL、加载 vLLM、批量评估、失败样本隔离、保存结果。
- 重新导出 data_io、metrics、vllm_runtime、outputs 中的核心函数，兼容测试导入。

使用方式：
- 通常通过 `python scripts/eval_binary_logits_qwen_omni_vllm.py ...` 调用。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

from omniav_detect.evaluation.constants import (  # noqa: F401
    LABELS,
    SYSTEM_PROMPT,
    USER_PROMPT,
    USER_PROMPT_AFTER_VIDEO,
)
from omniav_detect.evaluation.data_io import (  # noqa: F401
    batch_samples,
    load_jsonl_samples,
    normalize_video_path,
    write_json,
    write_jsonl,
)
from omniav_detect.evaluation.metrics import *  # noqa: F401,F403
from omniav_detect.evaluation.outputs import (  # noqa: F401
    build_run_config,
    make_bad_sample,
    print_core_metrics,
    save_outputs,
)
from omniav_detect.evaluation.vllm_runtime import *  # noqa: F401,F403
from omniav_detect.evaluation.vllm_runtime import evaluate_batch, evaluate_sample, load_vllm_engine
from omniav_detect.evaluation.visualization import write_eval_visualizations  # noqa: F401


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    函数功能：
    - 解析 vLLM 后端的单 checkpoint logits 评估命令行参数。

    参数：
    - argv: 命令行参数列表，None 时读取真实命令行。

    返回：
    - argparse.Namespace，包含模型、adapter、数据、batch 和 vLLM 参数。
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a Qwen2.5-Omni LoRA binary deepfake detector with vLLM logprobs."
    )
    parser.add_argument("--model_path", default="/data/OneDay/models/qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--fake_token_id", type=int, default=52317)
    parser.add_argument("--real_token_id", type=int, default=12768)
    parser.add_argument("--device_map", default="vllm", help="Kept for run logging; vLLM ignores this flag.")
    parser.add_argument("--torch_dtype", default="bfloat16", help="vLLM dtype name, e.g. bf16/float16.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--fps", type=float, default=1.0, help="Kept for logging; vLLM backend may ignore.")
    parser.add_argument(
        "--use_audio_in_video",
        dest="use_audio_in_video",
        action="store_true",
        help="Use the audio track embedded in each video. This is enabled by default.",
    )
    parser.add_argument(
        "--no_use_audio_in_video",
        dest="use_audio_in_video",
        action="store_false",
        help="Disable audio extraction from video during evaluation.",
    )
    parser.set_defaults(use_audio_in_video=True)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1,
        help="Number of tokens to generate (keep at 1 for logits evaluation).",
    )
    parser.add_argument("--save_every", type=int, default=100)

    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument(
        "--trust_remote_code",
        dest="trust_remote_code",
        action="store_true",
        help="Enable trust_remote_code for vLLM model loading.",
    )
    parser.add_argument(
        "--no_trust_remote_code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code for vLLM model loading.",
    )
    parser.set_defaults(trust_remote_code=True)
    parser.add_argument(
        "--mm_format",
        default="omni_av",
        choices=[
            "omni_av",
            "qwen_omni",
            "audio_video",
            "video",
            "qwen_vl",
            "qwen_vl_utils",
            "video_path",
            "video_dict",
            "videos_list",
            "video_audio",
            "none",
        ],
        help=(
            "Multi-modal input format passed to vLLM. 'omni_av' uses qwen_omni_utils/process_mm_info "
            "for audio-video input; 'video' uses qwen-vl-utils/process_vision_info for video-only input."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--logprobs",
        type=int,
        default=-1,
        help="Number of top logprobs to return; use -1 for full vocab (recommended for Real/Fake scoring).",
    )
    return parser.parse_args(argv)


def setup_logging() -> None:
    """初始化统一日志格式。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    函数功能：
    - vLLM 后端单 checkpoint 评估 CLI 主流程。

    参数：
    - argv: 命令行参数列表，None 时读取真实命令行。

    返回：
    - 进程退出码，0 表示运行完成。
    """
    args = parse_args(argv)
    setup_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_jsonl_samples(args.jsonl, args.max_samples)
    run_config = build_run_config(args, len(samples))
    run_config.update(
        {
            "backend": "vllm",
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "mm_format": args.mm_format,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "trust_remote_code": args.trust_remote_code,
        }
    )

    logging.info("Loaded %d samples from %s", len(samples), args.jsonl)
    logging.info("Output dir: %s", output_dir)
    logging.info("Fake token id=%d, Real token id=%d", args.fake_token_id, args.real_token_id)
    logging.info("Batch size=%d, use_audio_in_video=%s, fps=%s", args.batch_size, args.use_audio_in_video, args.fps)

    llm, tokenizer, lora_request = load_vllm_engine(args)
    predictions = []
    bad_samples = []
    processed = 0
    next_save = args.save_every if args.save_every > 0 else None

    for batch in batch_samples(samples, args.batch_size):
        try:
            predictions.extend(evaluate_batch(llm, tokenizer, lora_request, batch, args))
        except Exception as batch_exc:  # noqa: BLE001 - retry per sample to isolate broken media.
            if len(batch) > 1:
                logging.warning(
                    "Batch starting at sample %s failed with %s; retrying sample by sample.",
                    batch[0].get("index"),
                    batch_exc,
                )
            for sample in batch:
                try:
                    predictions.append(evaluate_sample(llm, tokenizer, lora_request, sample, args))
                except Exception as exc:  # noqa: BLE001 - continue evaluation after per-sample failures.
                    bad_samples.append(make_bad_sample(sample, exc))
                    logging.exception("Failed sample %s (%s)", sample.get("index"), sample.get("video_path"))

        processed += len(batch)
        if next_save is not None and processed >= next_save:
            save_outputs(output_dir, predictions, bad_samples, run_config)
            logging.info(
                "Saved checkpoint after %d/%d samples: %d predictions, %d bad samples",
                processed,
                len(samples),
                len(predictions),
                len(bad_samples),
            )
            while next_save is not None and next_save <= processed:
                next_save += args.save_every

    metrics = save_outputs(output_dir, predictions, bad_samples, run_config)
    logging.info("Wrote predictions.jsonl, bad_samples.jsonl, metrics.json, and visualizations to %s", output_dir)
    print_core_metrics(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
