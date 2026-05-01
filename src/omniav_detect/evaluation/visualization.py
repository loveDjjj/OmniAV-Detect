"""
本文件功能：
- 为 Qwen2.5-Omni binary 评估结果生成轻量可视化文件。

主要内容：
- write_eval_visualizations：写出混淆矩阵 CSV、分数分布 CSV 和 HTML 摘要。
- try_write_matplotlib_plots：如果环境有 matplotlib，额外生成 PNG 图。

使用方式：
- 被 `outputs.save_outputs` 调用；不强制依赖 matplotlib。
"""

from __future__ import annotations

import csv
import contextlib
import html
import io
from pathlib import Path
from typing import Any, Dict, List, Sequence


def write_eval_visualizations(
    output_dir: Path,
    metrics: Dict[str, Any],
    predictions: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    函数功能：
    - 写出评估指标的轻量可视化产物。

    参数：
    - output_dir: 单次评估输出目录。
    - metrics: `compute_metrics` 生成的指标字典。
    - predictions: 逐样本预测记录。

    返回：
    - 可视化目录和文件列表，供写入 metrics.json。

    关键逻辑：
    - 默认只依赖 Python 标准库；matplotlib 可用时补充 PNG。
    """
    visual_dir = output_dir / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)
    files: List[str] = []

    confusion_path = visual_dir / "confusion_matrix.csv"
    write_confusion_matrix_csv(confusion_path, metrics)
    files.append(str(confusion_path))

    score_path = visual_dir / "score_distribution.csv"
    write_score_distribution_csv(score_path, predictions)
    files.append(str(score_path))

    html_path = visual_dir / "summary.html"
    write_summary_html(html_path, metrics)
    files.append(str(html_path))

    files.extend(try_write_matplotlib_plots(visual_dir, metrics, predictions))
    return {"visualizations_dir": str(visual_dir), "visualization_files": files}


def write_confusion_matrix_csv(path: Path, metrics: Dict[str, Any]) -> None:
    """写出混淆矩阵 CSV，行是真值，列是预测。"""
    labels = metrics.get("confusion_matrix_labels", ["Fake", "Real"])
    matrix = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label\\prediction", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def write_score_distribution_csv(path: Path, predictions: Sequence[Dict[str, Any]]) -> None:
    """写出每条样本的 p_fake 分数，便于外部画图或排查阈值。"""
    fields = ["index", "label", "pred", "p_fake", "p_real", "video_path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in predictions:
            writer.writerow({field: item.get(field, "") for field in fields})


def pct(value: Any) -> str:
    """把浮点指标格式化为百分比文本。"""
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def write_summary_html(path: Path, metrics: Dict[str, Any]) -> None:
    """写出无需额外依赖即可打开的 HTML 指标摘要。"""
    rows = [
        ("Accuracy", pct(metrics.get("accuracy"))),
        ("AUC", "N/A" if metrics.get("roc_auc") is None else f"{float(metrics['roc_auc']):.4f}"),
        ("AP / mAP", "N/A" if metrics.get("average_precision") is None else f"{float(metrics['average_precision']):.4f}"),
        ("Fake recall", pct(metrics.get("fake_recall"))),
        ("Real recall", pct(metrics.get("real_recall"))),
        ("Predictions", str(metrics.get("num_predictions", 0))),
        ("Bad samples", str(metrics.get("num_bad_samples", 0))),
    ]
    labels = metrics.get("confusion_matrix_labels", ["Fake", "Real"])
    matrix = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    metric_rows = "\n".join(
        f"<tr><th>{html.escape(name)}</th><td>{html.escape(value)}</td></tr>" for name, value in rows
    )
    matrix_rows = "\n".join(
        "<tr>"
        f"<th>{html.escape(str(label))}</th>"
        + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row)
        + "</tr>"
        for label, row in zip(labels, matrix)
    )
    header_cells = "".join(f"<th>{html.escape(str(label))}</th>" for label in labels)
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Binary Deepfake Evaluation Summary</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    table {{ border-collapse: collapse; margin: 16px 0; min-width: 360px; }}
    th, td {{ border: 1px solid #ccc; padding: 8px 10px; text-align: right; }}
    th:first-child {{ text-align: left; background: #f5f5f5; }}
    caption {{ text-align: left; font-weight: 700; margin-bottom: 8px; }}
  </style>
</head>
<body>
  <h1>Binary Deepfake Evaluation Summary</h1>
  <table>
    <caption>Core Metrics</caption>
    {metric_rows}
  </table>
  <table>
    <caption>Confusion Matrix</caption>
    <tr><th>true\\pred</th>{header_cells}</tr>
    {matrix_rows}
  </table>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def try_write_matplotlib_plots(
    visual_dir: Path,
    metrics: Dict[str, Any],
    predictions: Sequence[Dict[str, Any]],
) -> List[str]:
    """matplotlib 可用时写 PNG；不可用时静默跳过，保证服务器最小环境可运行。"""
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []

    files: List[str] = []
    matrix_path = visual_dir / "confusion_matrix.png"
    labels = metrics.get("confusion_matrix_labels", ["Fake", "Real"])
    matrix = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    fig, ax = plt.subplots(figsize=(4, 3))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(matrix_path, dpi=160)
    plt.close(fig)
    files.append(str(matrix_path))

    scores = [float(item.get("p_fake", 0.0)) for item in predictions]
    if scores:
        score_path = visual_dir / "score_distribution.png"
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(scores, bins=20, color="#4c78a8", edgecolor="white")
        ax.set_xlabel("p_fake")
        ax.set_ylabel("Count")
        ax.set_title("Fake Score Distribution")
        fig.tight_layout()
        fig.savefig(score_path, dpi=160)
        plt.close(fig)
        files.append(str(score_path))
    return files
