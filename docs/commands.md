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

### 并行批量评估 dry-run

用途：读取 `configs/eval/qwen_omni_binary_batch_eval.yaml`，检查会展开成哪些并行评估命令。

```bash
python scripts/eval_batch_binary_qwen_omni.py \
  --config configs/eval/qwen_omni_binary_batch_eval.yaml \
  --dry_run
```

输入：

- `configs/eval/qwen_omni_binary_batch_eval.yaml`

输出：

- 控制台打印每个 run 的并行评估命令

备注：

- 默认后端为 `parallel`。
- 真实执行时，底层会调用内部并行脚本并按 worker 完成数显示进度。

### 并行批量正式评估

用途：按 YAML 配置评估默认三个 checkpoint，底层使用并行评估后端。

```bash
python scripts/eval_batch_binary_qwen_omni.py \
  --config configs/eval/qwen_omni_binary_batch_eval.yaml
```

输入：

- 批量评估 YAML
- 每个 run 对应的 JSONL 与 checkpoint

输出：

- 每个 run 的 `predictions.jsonl`、`bad_samples.jsonl`、`metrics.json`
- 每个 run 的 `visualizations/`
- 每个 run 的 `parallel_manifest.json`、`parallel_status.json`
- 每个 run 的 `workers/worker_*/`
- `batch_eval_summary.json`
- `batch_eval_summary.csv`

备注：

- 使用 `--only <run_name>` 可以只运行一个 run。
- 使用 `--batch_size`、`--fps`、`--max_samples` 可以临时覆盖 YAML 默认值。
- YAML 中的 `gpus` 和 `num_workers` 控制并行 worker 数。

### vLLM 批量评估 dry-run

用途：读取 `configs/eval/qwen_omni_binary_batch_eval_vllm.yaml`，检查会展开成哪些 vLLM 评估命令。

```bash
python scripts/eval_batch_binary_qwen_omni_vllm.py \
  --config configs/eval/qwen_omni_binary_batch_eval_vllm.yaml \
  --dry_run
```

输入：

- `configs/eval/qwen_omni_binary_batch_eval_vllm.yaml`

输出：

- 控制台打印每个 run 的 vLLM 评估命令

备注：

- 默认后端为 `vllm`。
- 真实执行时，底层 vLLM 评估会按 batch 显示进度。

### vLLM 批量正式评估

环境配置：
```bash
export VLLM_CU13_LIB=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH=$VLLM_CU13_LIB:$LD_LIBRARY_PATH
```
测试：
```bash
python -c "from vllm import LLM; print('vLLM import OK')"
```

用途：按 YAML 配置批量执行 vLLM 评估。

```bash
python scripts/eval_batch_binary_qwen_omni_vllm.py \
  --config configs/eval/qwen_omni_binary_batch_eval_vllm.yaml
```

输入：

- vLLM 批量评估 YAML
- 每个 run 对应的 JSONL 与 checkpoint

输出：

- 每个 run 的 `predictions.jsonl`、`bad_samples.jsonl`、`metrics.json`
- 每个 run 的 `visualizations/`
- `batch_eval_summary.json`
- `batch_eval_summary.csv`

备注：

- YAML 中的 `tensor_parallel_size`、`mm_format`、`logprobs` 控制 vLLM 评估行为。
- 如果只想跑纯视频 vLLM 路径，可将 YAML 中的 `mm_format` 改为 `video` 并关闭 `use_audio_in_video`。

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

### FakeAVCeleb 验证集视频时长随机抽样

用途：在服务器上从 `stage1_to_stage2_fakeavceleb_encoder_full/predictions.jsonl` 读取视频路径，随机抽样最多 1000 个视频，用 `ffprobe` 粗略统计时长分布。

```bash
python - <<'PY'
import json, random, subprocess, statistics
from pathlib import Path

pred_path = Path('/data/OneDay/OmniAV-Detect/outputs/batch_eval_binary/stage1_to_stage2_fakeavceleb_encoder_full/predictions.jsonl')
if not pred_path.exists():
    pred_path = Path('/data/OneDay/OmniAV-Detect/reports/batch_eval_binary/stage1_to_stage2_fakeavceleb_encoder_full/predictions.jsonl')

paths = []
with pred_path.open(encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        p = row.get('video_path') or row.get('video') or (row.get('videos') or [None])[0]
        if p:
            paths.append(p)

random.seed(42)
sample = random.sample(paths, min(1000, len(paths)))
durations = []
failed = []
for p in sample:
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        p,
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=20).strip()
        durations.append(float(out))
    except Exception as exc:
        failed.append({'path': p, 'error': str(exc)})

durations.sort()
def q(x):
    if not durations:
        return None
    idx = round((len(durations) - 1) * x)
    return durations[idx]

summary = {
    'source_predictions': str(pred_path),
    'sampled': len(sample),
    'ok': len(durations),
    'failed': len(failed),
    'min_sec': min(durations) if durations else None,
    'p25_sec': q(0.25),
    'median_sec': q(0.50),
    'p75_sec': q(0.75),
    'p90_sec': q(0.90),
    'p95_sec': q(0.95),
    'max_sec': max(durations) if durations else None,
    'mean_sec': statistics.mean(durations) if durations else None,
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
Path('reports/fakeavceleb_eval_duration_sample1000.json').write_text(json.dumps({'summary': summary, 'failed': failed[:50]}, ensure_ascii=False, indent=2), encoding='utf-8')
PY
```

输入：

- `reports/batch_eval_binary/stage1_to_stage2_fakeavceleb_encoder_full/predictions.jsonl` 或服务器 outputs 下同名文件
- `ffprobe` 可执行程序

输出：

- 控制台 JSON summary
- `reports/fakeavceleb_eval_duration_sample1000.json`

备注：

- 该命令只抽样读取视频元信息，不重新跑模型评估。
