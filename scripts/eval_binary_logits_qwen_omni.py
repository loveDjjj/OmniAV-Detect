#!/usr/bin/env python3
"""Evaluate a Qwen2.5-Omni LoRA binary Real/Fake detector from next-token logits."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SYSTEM_PROMPT = "You are an audio-video deepfake detector. Given an input video, answer only Real or Fake."
USER_PROMPT = "<video>\nGiven the video, please assess if it's Real or Fake? Only answer Real or Fake."
USER_PROMPT_AFTER_VIDEO = "Given the video, please assess if it's Real or Fake? Only answer Real or Fake."
LABELS = ["Fake", "Real"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Qwen2.5-Omni LoRA binary deepfake detector with Real/Fake token logits."
    )
    parser.add_argument("--model_path", default="/data/OneDay/models/qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--fake_token_id", type=int, default=52317)
    parser.add_argument("--real_token_id", type=int, default=12768)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--fps", type=float, default=1.0, help="Video sampling fps passed to the Qwen2.5-Omni processor.")
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
        help="Kept for run logging; logits evaluation uses one forward pass instead of generation.",
    )
    parser.add_argument("--save_every", type=int, default=100)
    return parser.parse_args(argv)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def build_conversation(video_path: str) -> List[Dict[str, Any]]:
    """Build the Qwen2.5-Omni chat format equivalent to the ms-swift video prompt."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": USER_PROMPT_AFTER_VIDEO},
            ],
        },
    ]


def normalize_video_path(raw_path: str, jsonl_path: Path) -> str:
    if raw_path.startswith("/"):
        return raw_path
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return str(path)
    return str((jsonl_path.parent / path).resolve())


def load_jsonl_samples(jsonl_path: Path | str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    path = Path(jsonl_path)
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_samples is not None and len(samples) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            videos = record.get("videos") or []
            meta = record.get("meta") or {}
            label = normalize_label(meta.get("overall_label"))
            if not videos:
                raise ValueError(f"{path}:{line_number} has no videos field")
            if label not in {"Real", "Fake"}:
                raise ValueError(f"{path}:{line_number} has invalid meta.overall_label={meta.get('overall_label')!r}")
            samples.append(
                {
                    "index": len(samples),
                    "line_number": line_number,
                    "video_path": normalize_video_path(str(videos[0]), path),
                    "label": label,
                    "meta": meta,
                    "source_record": record,
                }
            )
    return samples


def batch_samples(samples: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    for start in range(0, len(samples), batch_size):
        yield list(samples[start : start + batch_size])


def normalize_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text == "real":
        return "Real"
    if text == "fake":
        return "Fake"
    return str(value).strip()


def pair_softmax(real_logit: float, fake_logit: float) -> Dict[str, float | str]:
    max_logit = max(real_logit, fake_logit)
    real_exp = math.exp(real_logit - max_logit)
    fake_exp = math.exp(fake_logit - max_logit)
    denom = real_exp + fake_exp
    p_real = real_exp / denom
    p_fake = fake_exp / denom
    return {
        "real_logit": float(real_logit),
        "fake_logit": float(fake_logit),
        "p_real": float(p_real),
        "p_fake": float(p_fake),
        "pred": "Fake" if p_fake >= p_real else "Real",
    }


def resolve_torch_dtype(dtype_name: str, torch_module: Any) -> Any:
    normalized = str(dtype_name).strip().lower()
    if normalized in {"auto", "none"}:
        return "auto"
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "float": "float32",
        "fp32": "float32",
    }
    attr = aliases.get(normalized, normalized)
    if not hasattr(torch_module, attr):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return getattr(torch_module, attr)


def load_model_and_processor(args: argparse.Namespace) -> Tuple[Any, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependency. Please install transformers, peft, torch, and qwen_omni_utils "
            "in the Qwen2.5-Omni evaluation environment."
        ) from exc

    dtype = resolve_torch_dtype(args.torch_dtype, torch)
    logging.info("Loading base model from %s", args.model_path)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        enable_audio_output=False,
    )
    logging.info("Loading LoRA adapter from %s", args.adapter_path)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    return model, processor


def resolve_forward_model(model: Any) -> Any:
    """Return the module that owns Qwen2.5-Omni text logits forward.

    PeftModel.forward wraps the full Qwen2_5OmniForConditionalGeneration module. That full
    wrapper is a Thinker/Talker/Token2Wav composition, while the text logits forward lives in
    the Thinker. Calling the PEFT top-level forward can therefore fall through to
    torch.nn.Module._forward_unimplemented().
    """
    candidate_paths = [
        ("base_model", "model", "thinker"),
        ("model", "thinker"),
        ("thinker",),
    ]
    for path in candidate_paths:
        current = model
        for attr in path:
            current = getattr(current, attr, None)
            if current is None:
                break
        if current is not None:
            return current
    return model


def infer_input_device(model: Any) -> Any:
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return model_device
    for parameter in model.parameters():
        return parameter.device
    return None


def move_inputs_to_device(inputs: Any, device: Any) -> Any:
    if device is None:
        return inputs
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {key: move_inputs_to_device(value, device) for key, value in inputs.items()}
    if isinstance(inputs, list):
        return [move_inputs_to_device(value, device) for value in inputs]
    if isinstance(inputs, tuple):
        return tuple(move_inputs_to_device(value, device) for value in inputs)
    return inputs


def prepare_inputs(
    processor: Any,
    conversations: List[Dict[str, Any]] | List[List[Dict[str, Any]]],
    device: Any,
    use_audio_in_video: bool,
    fps: float,
) -> Any:
    try:
        from qwen_omni_utils import process_mm_info
    except ImportError as exc:
        try:
            inputs = processor.apply_chat_template(
                conversations,
                load_audio_from_video=use_audio_in_video,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                fps=fps,
                padding=True,
                use_audio_in_video=use_audio_in_video,
            )
            return move_inputs_to_device(inputs, device)
        except Exception as fallback_exc:  # noqa: BLE001 - preserve actionable dependency hint.
            raise RuntimeError(
                "qwen_omni_utils is unavailable and the Transformers built-in video decoder failed. "
                "Install qwen_omni_utils/decord or fix the torchvision video backend."
            ) from fallback_exc

    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=use_audio_in_video)
    common_kwargs = {
        "text": text,
        "images": images,
        "videos": videos,
        "return_tensors": "pt",
        "padding": True,
        "use_audio_in_video": use_audio_in_video,
    }
    try:
        inputs = processor(audio=audios, **common_kwargs)
    except TypeError:
        inputs = processor(audios=audios, **common_kwargs)
    return move_inputs_to_device(inputs, device)


def get_last_token_logits_batch(outputs: Any, inputs: Any) -> Any:
    logits = outputs.logits
    attention_mask = None
    if isinstance(inputs, dict):
        attention_mask = inputs.get("attention_mask")
    elif hasattr(inputs, "get"):
        attention_mask = inputs.get("attention_mask")

    if attention_mask is not None:
        last_indices = attention_mask.sum(dim=1).to(logits.device).long() - 1
        batch_indices = last_indices.new_tensor(range(logits.shape[0]))
        return logits[batch_indices, last_indices, :]
    return logits[:, -1, :]


def get_last_token_logits(outputs: Any, inputs: Any) -> Any:
    return get_last_token_logits_batch(outputs, inputs)[0]


def extract_binary_probs(logits: Any, real_token_id: int, fake_token_id: int) -> Dict[str, float | str]:
    vocab_size = int(logits.shape[-1])
    if real_token_id < 0 or real_token_id >= vocab_size:
        raise ValueError(f"real_token_id={real_token_id} is outside vocabulary size {vocab_size}")
    if fake_token_id < 0 or fake_token_id >= vocab_size:
        raise ValueError(f"fake_token_id={fake_token_id} is outside vocabulary size {vocab_size}")
    real_logit = float(logits[real_token_id].detach().float().cpu().item())
    fake_logit = float(logits[fake_token_id].detach().float().cpu().item())
    return pair_softmax(real_logit=real_logit, fake_logit=fake_logit)


def extract_binary_probs_batch(logits: Any, real_token_id: int, fake_token_id: int) -> List[Dict[str, float | str]]:
    if len(logits.shape) == 1:
        logits = logits.unsqueeze(0)
    return [
        extract_binary_probs(row_logits, real_token_id=real_token_id, fake_token_id=fake_token_id)
        for row_logits in logits
    ]


def evaluate_batch(
    model: Any,
    processor: Any,
    samples: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Missing torch in evaluation environment.") from exc

    forward_model = resolve_forward_model(model)
    conversations = [build_conversation(sample["video_path"]) for sample in samples]
    processor_input = conversations[0] if len(conversations) == 1 else conversations
    device = infer_input_device(forward_model)
    inputs = prepare_inputs(processor, processor_input, device, args.use_audio_in_video, args.fps)
    with torch.inference_mode():
        try:
            outputs = forward_model(**inputs, use_audio_in_video=args.use_audio_in_video)
        except TypeError:
            outputs = forward_model(**inputs)
    logits = get_last_token_logits_batch(outputs, inputs)
    score_rows = extract_binary_probs_batch(logits, real_token_id=args.real_token_id, fake_token_id=args.fake_token_id)
    records = []
    for sample, scores in zip(samples, score_rows):
        records.append(
            {
                "index": sample["index"],
                "line_number": sample["line_number"],
                "video_path": sample["video_path"],
                "label": sample["label"],
                "pred": scores["pred"],
                "p_real": scores["p_real"],
                "p_fake": scores["p_fake"],
                "real_logit": scores["real_logit"],
                "fake_logit": scores["fake_logit"],
                "meta": sample.get("meta", {}),
            }
        )
    return records


def evaluate_sample(model: Any, processor: Any, sample: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    return evaluate_batch(model, processor, [sample], args)[0]


def manual_confusion_matrix(labels: Sequence[str], predictions: Sequence[str]) -> List[List[int]]:
    index = {label: idx for idx, label in enumerate(LABELS)}
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    for label, prediction in zip(labels, predictions):
        if label in index and prediction in index:
            matrix[index[label]][index[prediction]] += 1
    return matrix


def average_ranks(values: Sequence[float]) -> List[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for idx in range(cursor, end):
            ranks[ordered[idx][0]] = average_rank
        cursor = end
    return ranks


def manual_roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = average_ranks(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, y_true) if label == 1)
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def manual_average_precision(y_true: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    positives = sum(y_true)
    if positives == 0:
        return None
    ordered = sorted(zip(scores, y_true), key=lambda item: item[0], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ordered, start=1):
        if label == 1:
            true_positives += 1
            precision_sum += true_positives / rank
    return float(precision_sum / positives)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def manual_classification_report(matrix: List[List[int]]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[idx][idx] for idx in range(len(LABELS)))
    per_label_rows = []
    for idx, label in enumerate(LABELS):
        tp = matrix[idx][idx]
        fp = sum(row[idx] for row in matrix) - tp
        fn = sum(matrix[idx]) - tp
        support = sum(matrix[idx])
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        row = {"precision": precision, "recall": recall, "f1-score": f1, "support": support}
        report[label] = row
        per_label_rows.append(row)

    report["accuracy"] = safe_div(correct, total)
    macro = {
        key: safe_div(sum(row[key] for row in per_label_rows), len(per_label_rows))
        for key in ["precision", "recall", "f1-score"]
    }
    macro["support"] = total
    weighted = {
        key: safe_div(sum(row[key] * row["support"] for row in per_label_rows), total)
        for key in ["precision", "recall", "f1-score"]
    }
    weighted["support"] = total
    report["macro avg"] = macro
    report["weighted avg"] = weighted
    return report


def try_import_sklearn_metrics() -> Tuple[Optional[Tuple[Any, Any, Any, Any]], Optional[str]]:
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            from sklearn.metrics import (
                average_precision_score,
                classification_report,
                confusion_matrix,
                roc_auc_score,
            )
        return (average_precision_score, classification_report, confusion_matrix, roc_auc_score), None
    except Exception as exc:  # noqa: BLE001 - sklearn can fail from binary dependency mismatches.
        details = str(exc)
        stderr_text = stderr.getvalue().strip()
        if stderr_text:
            details = f"{details}\n{stderr_text}"
        return None, details


def compute_metrics(predictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    labels = [normalize_label(item["label"]) for item in predictions]
    preds = [normalize_label(item["pred"]) for item in predictions]
    p_fake = [float(item["p_fake"]) for item in predictions]
    y_true = [1 if label == "Fake" else 0 for label in labels]
    accuracy = safe_div(sum(1 for label, pred in zip(labels, preds) if label == pred), len(labels))
    matrix = manual_confusion_matrix(labels, preds)

    metric_source = "manual"
    metric_errors: Dict[str, str] = {}
    roc_auc = manual_roc_auc(y_true, p_fake)
    average_precision = manual_average_precision(y_true, p_fake)
    classification = manual_classification_report(matrix)

    sklearn_metrics, sklearn_error = try_import_sklearn_metrics()
    if sklearn_metrics is not None:
        average_precision_score, classification_report, confusion_matrix, roc_auc_score = sklearn_metrics
        metric_source = "sklearn"
        matrix = confusion_matrix(labels, preds, labels=LABELS).tolist()
        classification = classification_report(labels, preds, labels=LABELS, output_dict=True, zero_division=0)
        try:
            roc_auc = float(roc_auc_score(y_true, p_fake))
        except ValueError as exc:
            metric_errors["roc_auc"] = str(exc)
            roc_auc = None
        try:
            average_precision = float(average_precision_score(y_true, p_fake))
        except ValueError as exc:
            metric_errors["average_precision"] = str(exc)
            average_precision = None
    else:
        metric_errors["sklearn"] = f"sklearn unavailable; manual metric fallback was used. {sklearn_error}"

    return {
        "num_predictions": len(predictions),
        "accuracy": float(accuracy),
        "roc_auc": roc_auc,
        "roc_auc_score": roc_auc,
        "average_precision": average_precision,
        "ap": average_precision,
        "map": average_precision,
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": matrix,
        "classification_report": classification,
        "label_distribution": dict(Counter(labels)),
        "prediction_distribution": dict(Counter(preds)),
        "metric_source": metric_source,
        "metric_errors": metric_errors,
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_outputs(
    output_dir: Path,
    predictions: Sequence[Dict[str, Any]],
    bad_samples: Sequence[Dict[str, Any]],
    run_config: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_jsonl(output_dir / "bad_samples.jsonl", bad_samples)
    metrics = compute_metrics(predictions) if predictions else {
        "num_predictions": 0,
        "accuracy": None,
        "roc_auc": None,
        "roc_auc_score": None,
        "average_precision": None,
        "ap": None,
        "map": None,
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": [[0, 0], [0, 0]],
        "classification_report": {},
        "label_distribution": {},
        "prediction_distribution": {},
        "metric_source": "none",
        "metric_errors": {},
    }
    metrics["num_bad_samples"] = len(bad_samples)
    metrics["run_config"] = run_config
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def make_bad_sample(sample: Dict[str, Any], error: BaseException) -> Dict[str, Any]:
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
    summary = {
        "num_predictions": metrics.get("num_predictions"),
        "num_bad_samples": metrics.get("num_bad_samples"),
        "accuracy": metrics.get("accuracy"),
        "roc_auc": metrics.get("roc_auc"),
        "average_precision": metrics.get("average_precision"),
        "label_distribution": metrics.get("label_distribution"),
        "prediction_distribution": metrics.get("prediction_distribution"),
        "confusion_matrix_labels": metrics.get("confusion_matrix_labels"),
        "confusion_matrix": metrics.get("confusion_matrix"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_jsonl_samples(args.jsonl, args.max_samples)
    run_config = build_run_config(args, len(samples))
    logging.info("Loaded %d samples from %s", len(samples), args.jsonl)
    logging.info("Output dir: %s", output_dir)
    logging.info("Fake token id=%d, Real token id=%d", args.fake_token_id, args.real_token_id)
    logging.info("Batch size=%d, use_audio_in_video=%s, fps=%s", args.batch_size, args.use_audio_in_video, args.fps)

    model, processor = load_model_and_processor(args)
    predictions: List[Dict[str, Any]] = []
    bad_samples: List[Dict[str, Any]] = []
    processed = 0
    next_save = args.save_every if args.save_every > 0 else None

    for batch in batch_samples(samples, args.batch_size):
        try:
            predictions.extend(evaluate_batch(model, processor, batch, args))
        except Exception as batch_exc:  # noqa: BLE001 - retry per sample to isolate broken media.
            if len(batch) > 1:
                logging.warning(
                    "Batch starting at sample %s failed with %s; retrying sample by sample.",
                    batch[0].get("index"),
                    batch_exc,
                )
            for sample in batch:
                try:
                    predictions.append(evaluate_sample(model, processor, sample, args))
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
    logging.info("Wrote predictions.jsonl, bad_samples.jsonl, and metrics.json to %s", output_dir)
    print_core_metrics(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
