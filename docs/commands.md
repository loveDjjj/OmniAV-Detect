# Commands

## 环境准备

### 安装依赖

用途：安装仓库中能确认需要的 Python 依赖。

```bash
pip install -r requirements.txt
```

输入：

- `requirements.txt`

输出：

- 当前 Python 环境中的依赖包

备注：

- conda 环境名和精确版本待确认。

## 数据准备

### FakeAVCeleb dry-run

用途：扫描 FakeAVCeleb，写统计和预览，不写训练 JSONL。

```bash
python scripts/prepare_swift_av_sft.py \
  --config configs/data/swift_av_sft.yaml \
  --dataset fakeavceleb \
  --dry_run \
  --num_preview 5
```

输入：

- 默认 `/data/OneDay/FakeAVCeleb`

输出：

- 默认 `/data/OneDay/OmniAV-Detect/data/swift_sft/fakeavceleb`
- `dataset_scan_summary.json`
- `dataset_stats.json`
- `missing_or_invalid_files.csv`
- `preview_samples.json`

备注：

- 正式生成 JSONL 时去掉 `--dry_run`，如需 structured 数据加 `--mode both`。

### MAVOS-DD dry-run

用途：扫描 MAVOS-DD，写统计和预览，不写训练 JSONL。

```bash
python scripts/prepare_swift_av_sft.py \
  --config configs/data/swift_av_sft.yaml \
  --dataset mavosdd \
  --dry_run \
  --num_preview 5
```

输入：

- 默认 `/data/OneDay/MAVOS-DD`

输出：

- 默认 `/data/OneDay/OmniAV-Detect/data/swift_sft/mavosdd`
- `dataset_scan_summary.json`
- `dataset_stats.json`
- `missing_or_invalid_files.csv`
- `preview_samples.json`

备注：

- 需要 `datasets` 读取本地 Arrow metadata。

## 评估 / 测试

### 单 checkpoint 小样本评估

用途：先跑 50 条样本，验证模型加载、视频解码和指标输出。

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_binary_logits_qwen_omni.py \
  --model_path /data/OneDay/models/qwen/Qwen2.5-Omni-7B \
  --adapter_path /data/OneDay/OmniAV-Detect/outputs/stage1_qwen2_5_omni_fakeavceleb_binary/checkpoint-944 \
  --jsonl /data/OneDay/OmniAV-Detect/data/swift_sft/fakeavceleb/fakeavceleb_binary_eval.jsonl \
  --output_dir /data/OneDay/OmniAV-Detect/outputs/eval_debug/fakeavceleb_stage1_50 \
  --batch_size 1 \
  --fps 1.0 \
  --max_samples 50 \
  --save_every 10
```

输入：

- Qwen2.5-Omni 基座模型
- LoRA checkpoint
- binary eval JSONL

输出：

- `predictions.jsonl`
- `bad_samples.jsonl`
- `metrics.json`
- `visualizations/confusion_matrix.csv`
- `visualizations/score_distribution.csv`
- `visualizations/summary.html`

备注：

- matplotlib 可用时会额外生成 PNG；不可用时仍会生成标准库实现的 CSV / HTML。

### 单 checkpoint vLLM 评估

用途：用 vLLM 后端跑 50 条样本，验证 logprob 评估流程。

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_binary_logits_qwen_omni_vllm.py \
  --model_path /data/OneDay/models/qwen/Qwen2.5-Omni-7B \
  --adapter_path /data/OneDay/OmniAV-Detect/outputs/stage1_qwen2_5_omni_fakeavceleb_binary/checkpoint-944 \
  --jsonl /data/OneDay/OmniAV-Detect/data/swift_sft/fakeavceleb/fakeavceleb_binary_eval.jsonl \
  --output_dir /data/OneDay/OmniAV-Detect/outputs/eval_debug/fakeavceleb_stage1_50_vllm \
  --batch_size 1 \
  --max_samples 50 \
  --mm_format video
```

输入：

- Qwen2.5-Omni 基座模型
- LoRA checkpoint
- binary eval JSONL

输出：

- `predictions.jsonl`
- `bad_samples.jsonl`
- `metrics.json`
- `visualizations/confusion_matrix.csv`
- `visualizations/score_distribution.csv`
- `visualizations/summary.html`

备注：

- 需要安装 vLLM；`mm_format` 需与 vLLM 的多模态接口一致。

### 批量评估 dry-run

用途：检查 YAML 配置会展开成哪些评估命令。

```bash
python scripts/eval_batch_binary_qwen_omni.py \
  --config configs/eval/qwen_omni_binary_batch_eval.yaml \
  --dry_run
```

输入：

- `configs/eval/qwen_omni_binary_batch_eval.yaml`

输出：

- 控制台打印每个 run 的单模型评估命令

### 批量正式评估

用途：按 YAML 配置评估默认三个 checkpoint。

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_batch_binary_qwen_omni.py \
  --config configs/eval/qwen_omni_binary_batch_eval.yaml
```

输入：

- 批量评估 YAML
- 每个 run 对应的 JSONL 与 checkpoint

输出：

- 每个 run 的 `predictions.jsonl`、`bad_samples.jsonl`、`metrics.json`
- 每个 run 的 `visualizations/`
- `batch_eval_summary.json`
- `batch_eval_summary.csv`

备注：

- 使用 `--only <run_name>` 可以只运行一个 run。
- 使用 `--batch_size`、`--fps`、`--max_samples` 可以临时覆盖 YAML 默认值。
- 使用 `--eval_script scripts/eval_binary_logits_qwen_omni_vllm.py` 可切换到 vLLM 后端。

## 常用检查命令

### 查看仓库关键文件

```bash
find . -maxdepth 3 -type f
```

PowerShell 可用：

```powershell
Get-ChildItem -File -Recurse -Depth 2 | Select-Object -ExpandProperty FullName
```

### 查看修改

```bash
git status
```
