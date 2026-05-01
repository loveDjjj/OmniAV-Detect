"""
本文件功能：
- 负责 Qwen2.5-Omni binary 评估结果、指标和可视化产物的写出。

主要内容：
- save_outputs：写 predictions/bad_samples/metrics，并生成 visualizations 目录。
- make_bad_sample：将异常样本转换为可写入 bad_samples.jsonl 的记录。
- build_run_config / print_core_metrics：保存和打印评估运行摘要。

使用方式：
- 被 `binary_logits.py` 主流程调用。
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict, Sequence

from omniav_detect.evaluation.constants import LABELS, SYSTEM_PROMPT, USER_PROMPT
from omniav_detect.evaluation.data_io import write_json, write_jsonl
from omniav_detect.evaluation.metrics import compute_metrics
from omniav_detect.evaluation.visualization import write_eval_visualizations


def empty_metrics() -> Dict[str, Any]:
    """没有成功预测时返回稳定的空指标结构。"""
    return {
        "num_predictions": 0,
        "accuracy": None,
        "roc_auc": None,
        "roc_auc_score": None,
        "average_precision": None,
        "ap": None,
        "map": None,
        "fake_recall": None,
        "real_recall": None,
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": [[0, 0], [0, 0]],
        "classification_report": {},
        "label_distribution": {},
        "prediction_distribution": {},
        "metric_source": "none",
        "metric_errors": {},
    }


def save_outputs(
    output_dir: Path,
    predictions: Sequence[Dict[str, Any]],
    bad_samples: Sequence[Dict[str, Any]],
    run_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    函数功能：
    - 写出预测、失败样本、指标和可视化文件。

    参数：
    - output_dir: 当前评估输出目录。
    - predictions: 成功预测记录。
    - bad_samples: 失败样本记录。
    - run_config: 本次评估参数摘要。

    返回：
    - metrics 字典，已包含可视化目录和文件列表。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_jsonl(output_dir / "bad_samples.jsonl", bad_samples)
    metrics = compute_metrics(predictions) if predictions else empty_metrics()
    metrics["num_bad_samples"] = len(bad_samples)
    metrics["run_config"] = run_config
    metrics.update(write_eval_visualizations(output_dir, metrics, predictions))
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def make_bad_sample(sample: Dict[str, Any], error: BaseException) -> Dict[str, Any]:
    """将单样本异常转换为 bad_samples.jsonl 中的一行。"""
    return {
        "index": sample.get("index"),
        "line_number": sample.get("line_number"),
        "video_path": sample.get("video_path"),
        "label": sample.get("label"),
        "reason": error.__class__.__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "meta": sample.get("meta", {}),
    }


def build_run_config(args: argparse.Namespace, num_samples: int) -> Dict[str, Any]:
    """保存本次评估的关键参数，写入 metrics.json 便于复现。"""
    return {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "jsonl": args.jsonl,
        "num_requested_samples": num_samples,
        "max_samples": args.max_samples,
        "fake_token_id": args.fake_token_id,
        "real_token_id": args.real_token_id,
        "device_map": args.device_map,
        "torch_dtype": args.torch_dtype,
        "batch_size": args.batch_size,
        "fps": args.fps,
        "use_audio_in_video": args.use_audio_in_video,
        "max_new_tokens": args.max_new_tokens,
        "save_every": args.save_every,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
    }


def print_core_metrics(metrics: Dict[str, Any]) -> None:
    """在控制台打印核心指标，避免用户必须打开 JSON 才能看到结果。"""
    summary = {
        "num_predictions": metrics.get("num_predictions"),
        "num_bad_samples": metrics.get("num_bad_samples"),
        "accuracy": metrics.get("accuracy"),
        "roc_auc": metrics.get("roc_auc"),
        "average_precision": metrics.get("average_precision"),
        "fake_recall": metrics.get("fake_recall"),
        "real_recall": metrics.get("real_recall"),
        "label_distribution": metrics.get("label_distribution"),
        "prediction_distribution": metrics.get("prediction_distribution"),
        "confusion_matrix_labels": metrics.get("confusion_matrix_labels"),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "visualizations_dir": metrics.get("visualizations_dir"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
