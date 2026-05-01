# OmniAV-Detect

## 项目简介

OmniAV-Detect 面向 audio-video deepfake detection，当前支持把 FakeAVCeleb 和 MAVOS-DD 本地数据转换为 ms-swift / Qwen2.5-Omni SFT JSONL，并评估 Qwen2.5-Omni LoRA binary detector 的 `Real` / `Fake` token logits。

## 项目结构

```text
configs/
  data/                       # 数据准备 YAML 配置
  eval/                       # 批量评估 YAML 配置
datasets/                     # 数据集下载辅助脚本
scripts/                      # 可直接运行的薄入口
src/omniav_detect/            # 核心 Python 包
  data/                       # 数据扫描、转换、统计
  evaluation/                 # 单模型评估和批量评估
tests/                        # 单元测试
docs/                         # 命令、架构、记录
```

## 关键文件说明

- `configs/data/swift_av_sft.yaml`：FakeAVCeleb / MAVOS-DD 数据准备路径和默认参数。
- `configs/eval/qwen_omni_binary_batch_eval.yaml`：三个 Qwen2.5-Omni LoRA checkpoint 的批量评估配置。
- `scripts/prepare_swift_av_sft.py`：统一数据准备入口，通过 `--dataset` 选择一个数据集。
- `scripts/eval_binary_logits_qwen_omni.py`：单 checkpoint logits 评估入口。
- `scripts/eval_batch_binary_qwen_omni.py`：批量评估入口。
- `src/omniav_detect/data/common.py`：视频扫描、ms-swift record 构造、统计文件写出。
- `src/omniav_detect/data/prepare_runner.py`：配置驱动的数据准备调度。
- `src/omniav_detect/data/fakeavceleb.py`：FakeAVCeleb metadata 合并、分层切分和输出构建。
- `src/omniav_detect/data/mavosdd.py`：MAVOS-DD Arrow metadata 读取和 open-set split 解析。
- `src/omniav_detect/evaluation/binary_logits.py`：单 checkpoint 评估主流程。
- `src/omniav_detect/evaluation/model_runtime.py`：Qwen2.5-Omni + LoRA 加载、多模态输入处理和 logits forward。
- `src/omniav_detect/evaluation/metrics.py`：Accuracy、AUC、AP/mAP、Confusion Matrix 和 recall 指标计算。
- `src/omniav_detect/evaluation/visualization.py`：评估结果的 CSV / HTML 可视化输出，matplotlib 可用时额外生成 PNG。
- `src/omniav_detect/evaluation/batch_runner.py`：按 YAML 配置顺序调度多个评估任务。

## 数据说明

数据准备输入为本地视频文件，不复制视频；输出 JSONL 中的 `videos` 保存绝对路径。每条 SFT 记录包含 `messages`、`videos` 和 `meta`，binary SFT 的 assistant 只回答 `Real` 或 `Fake`。

数据准备会同时写出 `dataset_scan_summary.json`、`dataset_stats.json`、`missing_or_invalid_files.csv` 和 `preview_samples.json`。评估会输出 `predictions.jsonl`、`bad_samples.jsonl`、`metrics.json` 和 `visualizations/`，批量评估会额外输出 `batch_eval_summary.json/csv`。

## 运行方式

FakeAVCeleb dry-run：

```bash
python scripts/prepare_swift_av_sft.py --dataset fakeavceleb --dry_run --num_preview 5
```

MAVOS-DD dry-run：

```bash
python scripts/prepare_swift_av_sft.py --dataset mavosdd --dry_run --num_preview 5
```

批量评估 dry-run：

```bash
python scripts/eval_batch_binary_qwen_omni.py \
  --config configs/eval/qwen_omni_binary_batch_eval.yaml \
  --dry_run
```

详细命令见 `docs/commands.md`。

## 阅读顺序

1. `configs/data/swift_av_sft.yaml`
2. `configs/eval/qwen_omni_binary_batch_eval.yaml`
3. `src/omniav_detect/data/prepare_runner.py`
4. `src/omniav_detect/data/common.py`
5. `src/omniav_detect/data/fakeavceleb.py` 和 `src/omniav_detect/data/mavosdd.py`
6. `src/omniav_detect/evaluation/binary_logits.py`
7. `src/omniav_detect/evaluation/model_runtime.py`、`metrics.py`、`outputs.py`、`visualization.py`
8. `src/omniav_detect/evaluation/batch_runner.py`
