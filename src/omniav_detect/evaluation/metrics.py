"""
本文件功能：
- 计算 Qwen2.5-Omni binary deepfake 评估指标。

主要内容：
- pair_softmax：在 Real/Fake 两个 token logits 上做二分类 softmax。
- compute_metrics：计算 Accuracy、AUC、AP/mAP、Confusion Matrix、Fake recall、Real recall。
- manual_*：在 sklearn/scipy 不可用时提供手写指标后备实现。

使用方式：
- 被 `outputs.save_outputs` 和测试直接调用。
"""

from __future__ import annotations

import contextlib
import io
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from omniav_detect.evaluation.constants import LABELS


def normalize_label(value: Any) -> str:
    """将各种输入标签规范化为 `Real` / `Fake`。"""
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text == "real":
        return "Real"
    if text == "fake":
        return "Fake"
    return str(value).strip()


def pair_softmax(real_logit: float, fake_logit: float) -> Dict[str, float | str]:
    """
    函数功能：
    - 对 Real/Fake 两个 token logit 做稳定 softmax。

    参数：
    - real_logit: Real token 的 logit。
    - fake_logit: Fake token 的 logit。

    返回：
    - 概率、原始 logit 和预测标签。
    """
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


def manual_confusion_matrix(labels: Sequence[str], predictions: Sequence[str]) -> List[List[int]]:
    """按 `LABELS = [Fake, Real]` 计算 2x2 混淆矩阵。"""
    index = {label: idx for idx, label in enumerate(LABELS)}
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    for label, prediction in zip(labels, predictions):
        if label in index and prediction in index:
            matrix[index[label]][index[prediction]] += 1
    return matrix


def average_ranks(values: Sequence[float]) -> List[float]:
    """计算带并列值平均名次的 rank，用于手写 AUC。"""
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
    """以 Fake=1、p_fake 为 score 手写 ROC-AUC。"""
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = average_ranks(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, y_true) if label == 1)
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def manual_average_precision(y_true: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """以 Fake=1、p_fake 为 score 手写 Average Precision。"""
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
    """除法安全封装，分母为 0 时返回 0。"""
    return float(numerator / denominator) if denominator else 0.0


def manual_classification_report(matrix: List[List[int]]) -> Dict[str, Any]:
    """根据混淆矩阵生成 sklearn 风格的分类报告字典。"""
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
    """
    函数功能：
    - 尝试导入 sklearn.metrics，并捕获 ABI 等非 ImportError 问题。

    返回：
    - 成功时返回四个 sklearn 函数；失败时返回错误文本。
    """
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
    """
    函数功能：
    - 基于预测结果计算二分类评估指标。

    参数：
    - predictions: 模型预测记录列表。

    返回：
    - 指标字典，包含 Accuracy、AUC、AP/mAP、Confusion Matrix、Fake recall、Real recall。

    关键逻辑：
    - Fake 作为正类；sklearn 不可用时使用手写指标作为后备。
    """
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

    fake_recall = safe_div(matrix[0][0], sum(matrix[0]))
    real_recall = safe_div(matrix[1][1], sum(matrix[1]))
    return {
        "num_predictions": len(predictions),
        "accuracy": float(accuracy),
        "roc_auc": roc_auc,
        "roc_auc_score": roc_auc,
        "average_precision": average_precision,
        "ap": average_precision,
        "map": average_precision,
        "fake_recall": fake_recall,
        "real_recall": real_recall,
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": matrix,
        "classification_report": classification,
        "label_distribution": dict(Counter(labels)),
        "prediction_distribution": dict(Counter(preds)),
        "metric_source": metric_source,
        "metric_errors": metric_errors,
    }
