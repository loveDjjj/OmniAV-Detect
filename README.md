# OmniAV-Detect

## 项目简介

OmniAV-Detect 面向 audio-video deepfake detection，当前支持把 FakeAVCeleb 和 MAVOS-DD 本地数据转换为 ms-swift / Qwen2.5-Omni SFT JSONL，并通过两条批量评估路径计算 Qwen2.5-Omni 基模或 LoRA binary detector 的 `Real` / `Fake` token logits：并行评估后端和 vLLM 后端。仓库根目录的 `mvad/` 额外提供 MVAD 公开 train 数据的独立预处理和 Qwen2.5-Omni baseline 训练脚本。

其中 FakeAVCeleb 当前支持两种数据划分 protocol：

- 默认 `70% / 30%` 的 `overall_label + modality_type` 分层切分。
- 参考 MRDF 的 subject-independent `5-fold` 划分，读取 `data_fakeavceleb/train_*.txt` 和 `test_*.txt`。

## 项目结构

```text
configs/
  data/                       # 数据准备 YAML 配置
  eval/                       # 批量评估 YAML 配置
datasets/                     # 数据集下载辅助脚本
mvad/                         # MVAD 解压、划分、抽音频和 JSONL 生成脚本
scripts/                      # 可直接运行的薄入口
src/omniav_detect/            # 核心 Python 包
  data/                       # 数据扫描、转换、统计
  evaluation/                 # 单模型评估和批量评估
tests/                        # 单元测试
docs/                         # 命令、架构、记录
```

## 关键文件说明

- `configs/data/swift_av_sft.yaml`：FakeAVCeleb / MAVOS-DD 数据准备路径和默认参数。
- `configs/eval/qwen_omni_binary_batch_eval.yaml`：并行评估后端的批量评估配置。
- `configs/eval/qwen_omni_binary_batch_eval_vllm.yaml`：vLLM 后端的批量评估配置。
- `scripts/prepare_swift_av_sft.py`：统一数据准备入口，通过 `--dataset` 选择一个数据集。
- `scripts/eval_batch_binary_qwen_omni.py`：并行评估后端的批量评估入口。
- `scripts/eval_batch_binary_qwen_omni_vllm.py`：vLLM 后端的批量评估入口。
- `src/omniav_detect/data/common.py`：视频扫描、ms-swift record 构造、统计文件写出。
- `src/omniav_detect/data/prepare_runner.py`：配置驱动的数据准备调度。
- `src/omniav_detect/data/fakeavceleb.py`：FakeAVCeleb metadata 合并、默认分层切分、MRDF 5 折切分和输出构建。
- `src/omniav_detect/data/mavosdd.py`：MAVOS-DD Arrow metadata 读取和 open-set split 解析。
- `scripts/extract_audio_and_build_av_jsonl.py`：从已有视频 JSONL 批量抽取音频，并生成带 `audios` 字段的新 JSONL。
- `mvad/prepare_mvad.py`：MVAD 专用一体化预处理入口，生成 internal train/val 显式音频 JSONL。
- `mvad/train_stage1_MVAD.sh`：MVAD 显式音频 stage1 LoRA baseline 训练脚本。
- `src/omniav_detect/evaluation/batch_runner.py`：按 YAML 调度并行或 vLLM 批量评估。
- `src/omniav_detect/evaluation/parallel_runner.py`：多 GPU JSONL 分片、worker 调度、预测合并和指标重算。
- `src/omniav_detect/evaluation/binary_logits_vllm.py`：vLLM 后端单次评估主流程，由批量入口调用。
- `src/omniav_detect/evaluation/vllm_runtime.py`：vLLM 推理、多模态输入和 logprob 解析。
- `src/omniav_detect/evaluation/metrics.py`：Accuracy、AUC、AP/mAP、Confusion Matrix 和 recall 指标计算。
- `src/omniav_detect/evaluation/visualization.py`：评估结果的 CSV / HTML 可视化输出，matplotlib 可用时额外生成 PNG。

## 数据说明

数据准备输入为本地视频文件，不复制视频；输出 JSONL 中的 `videos` 保存绝对路径。每条 SFT 记录包含 `messages`、`videos` 和 `meta`，binary SFT 的 assistant 只回答 `Real` 或 `Fake`。

如果需要显式把音频作为独立输入喂给 Qwen 多模态模型，可在生成视频 JSONL 后，再运行 `scripts/extract_audio_and_build_av_jsonl.py` 额外写出带 `audios` 字段的数据集。

注意：

- 旧流程是 `videos` + `use_audio_in_video=True`，由框架从视频里自动抽音频。
- 新流程是 `videos + audios`，训练时必须关闭 `use_audio_in_video`，否则同一份音频会重复进入模型。

数据准备会同时写出 `dataset_scan_summary.json`、`dataset_stats.json`、`missing_or_invalid_files.csv` 和 `preview_samples.json`。评估会输出 `predictions.jsonl`、`bad_samples.jsonl`、`metrics.json` 和 `visualizations/`，批量评估会额外输出 `batch_eval_summary.json/csv`。

## 运行方式

FakeAVCeleb dry-run：

```bash
python scripts/prepare_swift_av_sft.py --dataset fakeavceleb --dry_run --num_preview 5
```

FakeAVCeleb MRDF 5-fold dry-run：

```bash
python scripts/prepare_swift_av_sft.py --dataset fakeavceleb_mrdf5fold --dry_run --num_preview 5
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

vLLM 批量评估 dry-run：

```bash
python scripts/eval_batch_binary_qwen_omni_vllm.py \
  --config configs/eval/qwen_omni_binary_batch_eval_vllm.yaml \
  --dry_run
```

详细命令见 `docs/commands.md`。

## 阅读顺序

1. `configs/data/swift_av_sft.yaml`
2. `configs/eval/qwen_omni_binary_batch_eval.yaml`
3. `src/omniav_detect/data/prepare_runner.py`
4. `src/omniav_detect/data/common.py`
5. `src/omniav_detect/data/fakeavceleb.py` 和 `src/omniav_detect/data/mavosdd.py`
6. `src/omniav_detect/evaluation/batch_runner.py`
7. `src/omniav_detect/evaluation/parallel_runner.py`
8. `src/omniav_detect/evaluation/binary_logits_vllm.py`、`vllm_runtime.py`、`metrics.py`、`outputs.py`、`visualization.py`
