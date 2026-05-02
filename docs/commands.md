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
- 不传 `--adapter_path` 时会直接评估未微调的 Qwen2.5-Omni 基模，适合 0-shot 对照实验。

### 单 checkpoint vLLM 评估

用途：用 vLLM 后端跑 50 条样本，验证 audio-video 输入和 Real/Fake logprob 评估流程。

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_binary_logits_qwen_omni_vllm.py \
  --model_path /data/OneDay/models/qwen/Qwen2.5-Omni-7B \
  --adapter_path /data/OneDay/OmniAV-Detect/outputs/stage1_qwen2_5_omni_fakeavceleb_binary/checkpoint-944 \
  --jsonl /data/OneDay/OmniAV-Detect/data/swift_sft/fakeavceleb/fakeavceleb_binary_eval.jsonl \
  --output_dir /data/OneDay/OmniAV-Detect/outputs/eval_debug/fakeavceleb_stage1_50_vllm \
  --batch_size 1 \
  --max_samples 50 \
  --mm_format omni_av \
  --logprobs -1
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

- 需要安装 vLLM 与 `qwen-omni-utils`；默认 `mm_format=omni_av` 会调用 `process_mm_info` 构造 audio+video 输入。
- `--logprobs -1` 会请求完整词表 logprobs；如果 vLLM 版本支持 `logprob_token_ids`，代码会自动只请求 Real/Fake 两个 token。
- 如果只想跑纯视频 vLLM 路径，可使用 `--mm_format video --no_use_audio_in_video`，此时需要 `qwen-vl-utils`。

### 单 checkpoint 多 GPU 并行评估

用途：把一个 eval JSONL 切成多个 shard，每张 GPU 启动一个独立评估子进程，最后合并预测并重新计算整体指标。

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_parallel_binary_qwen_omni.py \
  --model_path /data/OneDay/models/qwen/Qwen2.5-Omni-7B \
  --adapter_path /data/OneDay/OmniAV-Detect/outputs/stage1_qwen2_5_omni_fakeavceleb_binary/checkpoint-944 \
  --jsonl /data/OneDay/OmniAV-Detect/data/swift_sft/fakeavceleb/fakeavceleb_binary_eval.jsonl \
  --output_dir /data/OneDay/OmniAV-Detect/outputs/eval_parallel/fakeavceleb_stage1 \
  --gpus 0,1 \
  --num_workers 2 \
  --batch_size 1 \
  --fps 1.0 \
  --save_every 100
```

输入：

- Qwen2.5-Omni 基座模型
- LoRA checkpoint
- binary eval JSONL

输出：

- 合并后的 `predictions.jsonl`
- 合并后的 `bad_samples.jsonl`
- 重算后的 `metrics.json`
- `visualizations/`
- `parallel_manifest.json`
- `parallel_status.json`
- 每个 worker 的 `workers/worker_*/`

备注：

- `--num_workers` 表示 GPU worker 进程数，不是 PyTorch DataLoader 的数据加载线程数。
- 每个 worker 会独立加载一份模型；2 张 48G GPU 通常设置 `--gpus 0,1 --num_workers 2`。
- 分片使用 round-robin，适合 FakeAVCeleb 这类可能按类别排序的 JSONL。
- 默认不保留临时 `shards/`；需要排查 shard 内容时加 `--keep_shards`。
- 不传 `--adapter_path` 时会并行评估未微调基模，输出格式与 LoRA 评估完全一致。

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

## 训练

### FakeAVCeleb: stage1 产物接 stage2 思路继续训练

用途：先将 FakeAVCeleb 的 stage1 LoRA checkpoint merge 到基模，再按当前 stage2 的 `full + freeze_llm` 思路继续训练。

```bash
bash train_stage1_to_stage2_FakeAVCeleb.sh
```

输入：

- `/data/OneDay/OmniAV-Detect/outputs/stage1_qwen2_5_omni_fakeavceleb_binary_lora_audio_in_video/checkpoint-*`
- `/data/OneDay/OmniAV-Detect/data/swift_sft/fakeavceleb/fakeavceleb_binary_train.jsonl`

输出：

- merge 后模型目录：`/data/OneDay/OmniAV-Detect/outputs/stage1_to_stage2_fakeavceleb_merged`
- stage2 继续训练输出：`/data/OneDay/OmniAV-Detect/outputs/stage1_to_stage2_fakeavceleb_encoder_full`

备注：

- 脚本会自动选择 `stage1` 输出目录下最新的 `checkpoint-*`。
- 训练逻辑沿用当前 `train_stage2_FakeAVCeleb.sh` 的冻结策略和学习率。

### MAVOS-DD: stage1 产物接 stage2 思路继续训练

用途：先将 MAVOS-DD 的 stage1 LoRA checkpoint merge 到基模，再按当前 stage2 的 `full + freeze_llm` 思路继续训练。

```bash
bash train_stage1_to_stage2_MAVOS-DD.sh
```

输入：

- `/data/OneDay/OmniAV-Detect/outputs/stage1_qwen2_5_omni_mavosdd_binary_audio_in_video/checkpoint-*`
- `/data/OneDay/OmniAV-Detect/data/swift_sft/mavosdd/mavosdd_binary_train.jsonl`

输出：

- merge 后模型目录：`/data/OneDay/OmniAV-Detect/outputs/stage1_to_stage2_mavosdd_merged`
- stage2 继续训练输出：`/data/OneDay/OmniAV-Detect/outputs/stage1_to_stage2_mavosdd_encoder_full`

备注：

- 脚本会自动选择 `stage1` 输出目录下最新的 `checkpoint-*`。
- 训练逻辑沿用当前 `train_stage2_MAVOS-DD.sh` 的冻结策略、像素约束和梯度检查点设置。

### 顺序执行两个数据集的 stage1 -> stage2 训练

用途：依次运行 FakeAVCeleb 和 MAVOS-DD 的 stage1 merge + stage2 训练脚本，并把日志写入 `logs/`。

```bash
bash train_all_stage1_stage2.sh
```

输入：

- 上述两个单数据集脚本的默认路径

输出：

- `logs/train_stage1_to_stage2_FakeAVCeleb.log`
- `logs/train_stage1_to_stage2_MAVOS-DD.log`

备注：

- 不会覆盖你现有的 `train_all.sh`。

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
