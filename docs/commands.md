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

### FakeAVCeleb MRDF 5-fold dry-run

用途：按 MRDF 的 subject-independent 5 折 protocol 生成 FakeAVCeleb train/test JSONL 定义，并写统计与预览。

```bash
python scripts/prepare_swift_av_sft.py \
  --config configs/data/swift_av_sft.yaml \
  --dataset fakeavceleb_mrdf5fold \
  --dry_run \
  --num_preview 5
```

输入：

- 默认 `/data/OneDay/FakeAVCeleb`
- 默认 `/data/OneDay/OmniAV-Detect/data_fakeavceleb/train_*.txt`
- 默认 `/data/OneDay/OmniAV-Detect/data_fakeavceleb/test_*.txt`

输出：

- 默认 `/data/OneDay/OmniAV-Detect/data/swift_sft/fakeavceleb_mrdf5fold`
- `fakeavceleb_mrdf5fold_fold*_binary_train.jsonl`
- `fakeavceleb_mrdf5fold_fold*_binary_test.jsonl`
- `dataset_scan_summary.json`
- `dataset_stats.json`
- `missing_or_invalid_files.csv`
- `preview_samples.json`

备注：

- 当前实现按 metadata 的 `source` 字段或目录中的 `idxxxxx` subject id 匹配 5 折文件。
- 如果 subject id 无法解析，脚本会直接报错，避免静默生成错误划分。

### 从视频 JSONL 抽取音频并生成带 audios 字段的新 JSONL

用途：从已有 FakeAVCeleb / MAVOS-DD 视频 JSONL 中批量抽取音频，生成显式包含 `audios` 字段的 AV 数据集。

```bash
python scripts/extract_audio_and_build_av_jsonl.py \
  --input_jsonl /data/OneDay/OmniAV-Detect/data/swift_sft/fakeavceleb_mrdf5fold/fakeavceleb_mrdf5fold_fold1_binary_train.jsonl \
  --output_jsonl /data/OneDay/OmniAV-Detect/data/swift_sft/fakeavceleb_mrdf5fold/fakeavceleb_mrdf5fold_fold1_binary_train_with_audio.jsonl \
  --audio_root /data/OneDay/OmniAV-Detect/data/audio_cache/fakeavceleb_mrdf5fold \
  --sample_rate 16000 \
  --audio_channels 1
```

输入：

- 已生成的 JSONL，要求每条记录有 `videos[0]`
- `ffmpeg` 可执行程序

输出：

- 新 JSONL：保留原 `messages` / `videos` / `meta`，新增 `audios`
- 音频文件目录：默认按视频原路径层级写到 `audio_root`

备注：

- 脚本默认输出 `.wav`，并使用单声道 16 kHz PCM。
- 可先加 `--dry_run` 只检查路径组织是否符合预期。

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

### MVAD 公开 train 数据一键预处理

用途：解压 MVAD zip，按 group-aware 规则划分 internal train/val，抽取音频，并生成 Qwen2.5-Omni 显式 `audios` JSONL。

```bash
SOURCE_ROOT=/data/MVAD bash mvad/run_prepare_mvad.sh
```

输入：

- 默认 `/data/MVAD`
- 目录下应包含 `train/real_real`、`train/real_fake`、`train/fake_real`、`train/fake_fake`

输出：

- `/data/OneDay/OmniAV-Detect/data/mvad_unpacked`
- `/data/OneDay/OmniAV-Detect/data/mvad_processed/mvad_train_index.jsonl`
- `/data/OneDay/OmniAV-Detect/data/mvad_processed/mvad_val_index.jsonl`
- `/data/OneDay/OmniAV-Detect/data/mvad_processed/split_stats.json`
- `/data/OneDay/OmniAV-Detect/data/audio_cache/mvad`
- `/data/OneDay/OmniAV-Detect/data/swift_sft/mvad/mvad_binary_train_with_audio.jsonl`
- `/data/OneDay/OmniAV-Detect/data/swift_sft/mvad/mvad_binary_val_with_audio.jsonl`

备注：

- 当前 MVAD 公开数据只有 train，因此这是 internal validation baseline，不是论文官方 test 复现。
- 默认使用 `7z x` 解压 zip；如果 7z 命令名不同，可设置 `EXTRACTOR=7za` 或 `EXTRACTOR=7zz`。
- 解压和抽音频阶段会显示进度条。
- JSONL 已显式包含 `audios`，后续训练必须关闭 `use_audio_in_video`。
- 如果 zip 已经解压，可用 `SKIP_UNZIP=true`；如果音频已抽取，可用 `SKIP_AUDIO=true`。
- 如果个别 zip 损坏，可用 `SKIP_BAD_ARCHIVES=true`，将坏包记录到 `mvad_processed/unpack_manifest.json` 并继续处理其他压缩包。
- 扫描解压视频时会跳过 macOS zip 常见的 `__MACOSX` 和 `._*` 资源叉文件，避免把伪 `.mp4` 送入 ffmpeg。

中断后续跑：

```bash
SOURCE_ROOT=/data/OneDay/MVAD SKIP_UNZIP=true bash mvad/run_prepare_mvad.sh
```

用途：

- 解压已经完成、抽音频中途因坏样本或中断退出时，从现有解压目录重新构建 index 并继续抽音频。
- 已经存在的 wav 默认不会重复抽取；只会补齐缺失音频并重写 JSONL。

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

### FakeAVCeleb MRDF 5-fold + 显式 audios: stage1 训练

用途：使用 MRDF 5 折 train JSONL 和显式 `audios` 字段，做 FakeAVCeleb 的 stage1 LoRA 训练。

```bash
FOLD_ID=1 bash train_stage1_FakeAVCeleb_MRDF5Fold_Audio.sh
```

输入：

- `/data/OneDay/OmniAV-Detect/data/swift_sft/fakeavceleb_mrdf5fold/fakeavceleb_mrdf5fold_fold1_binary_train_with_audio.jsonl`

输出：

- `/data/OneDay/OmniAV-Detect/outputs/stage1_qwen2_5_omni_fakeavceleb_mrdf5fold_fold1_binary_audio_explicit`

备注：

- 当前脚本显式设置 `USE_AUDIO_IN_VIDEO=False`，避免视频内音频和 `audios` 字段重复进入模型。
- 切换 fold 时只需修改 `FOLD_ID=1..5`。

### FakeAVCeleb MRDF 5-fold + 显式 audios: stage1 产物接 stage2 思路继续训练

用途：先将 MRDF 5 折的 stage1 LoRA checkpoint merge 到基模，再按 stage2 的 `full + freeze_llm` 思路继续训练。

```bash
FOLD_ID=1 bash train_stage1_to_stage2_FakeAVCeleb_MRDF5Fold_Audio.sh
```

输入：

- `/data/OneDay/OmniAV-Detect/outputs/stage1_qwen2_5_omni_fakeavceleb_mrdf5fold_fold1_binary_audio_explicit/checkpoint-*`
- `/data/OneDay/OmniAV-Detect/data/swift_sft/fakeavceleb_mrdf5fold/fakeavceleb_mrdf5fold_fold1_binary_train_with_audio.jsonl`

输出：

- merge 后模型目录：`/data/OneDay/OmniAV-Detect/outputs/stage1_to_stage2_fakeavceleb_mrdf5fold_fold1_merged`
- stage2 输出：`/data/OneDay/OmniAV-Detect/outputs/stage1_to_stage2_fakeavceleb_mrdf5fold_fold1_encoder_full_audio_explicit`

备注：

- 当前脚本同样显式设置 `USE_AUDIO_IN_VIDEO=False`。
- 默认会自动寻找 `STAGE1_OUTPUT_DIR` 下最新的 `checkpoint-*`。
- 如果数据集、输出目录或 fold 编号不同，可通过环境变量覆盖。

### FakeAVCeleb MRDF 5-fold + 显式 audios: stage2 only 训练

用途：不接 stage1，直接从 Qwen2.5-Omni 基模运行 MRDF 5 折的 Stage 2 only full tuning。

```bash
FOLD_ID=1 bash train_stage2_FakeAVCeleb_MRDF5Fold_Audio.sh
```

输入：

- `/data/OneDay/OmniAV-Detect/data/swift_sft/fakeavceleb_mrdf5fold/fakeavceleb_mrdf5fold_fold1_binary_train_with_audio.jsonl`

输出：

- `/data/OneDay/OmniAV-Detect/outputs/stage2_qwen2_5_omni_fakeavceleb_mrdf5fold_fold1_encoder_full_audio_explicit`

备注：

- 训练策略参考 `train_stage2_FakeAVCeleb.sh`：`tuner_type=full`、`freeze_llm=true`、`freeze_vit=false`、`freeze_aligner=false`。
- 当前脚本显式设置 `USE_AUDIO_IN_VIDEO=False`，避免视频内音频和 `audios` 字段重复进入模型。
- 切换 fold 时修改 `FOLD_ID=1..5`。

### FakeAVCeleb MRDF 5-fold + 显式 audios: fold1 stage1 评估

用途：评估 fold1 的 stage1 LoRA 模型，输入 JSONL 已包含 `audios` 字段。

```bash
bash eval_fakeavceleb_mrdf5fold_fold1_stage1_audio.sh
```

输入：

- `/data/OneDay/OmniAV-Detect/outputs/stage1_qwen2_5_omni_fakeavceleb_mrdf5fold_fold1_binary_audio_explicit/checkpoint-*`
- `/data/OneDay/OmniAV-Detect/data/swift_sft/fakeavceleb_mrdf5fold/fakeavceleb_mrdf5fold_fold1_binary_test_with_audio.jsonl`

输出：

- `/data/OneDay/OmniAV-Detect/outputs/batch_eval_binary/fakeavceleb_mrdf5fold_fold1_stage1_audio`

备注：

- 脚本会自动寻找 stage1 输出目录下最新的 `checkpoint-*`。
- 脚本强制关闭 `use_audio_in_video`，避免 `audios` 字段和视频内音频重复输入。

### FakeAVCeleb MRDF 5-fold + 显式 audios: fold1 stage1->stage2 评估

用途：评估 fold1 的 stage1->stage2 full checkpoint，输入 JSONL 已包含 `audios` 字段。

```bash
bash eval_fakeavceleb_mrdf5fold_fold1_stage1_to_stage2_audio.sh
```

输入：

- `/data/OneDay/OmniAV-Detect/outputs/stage1_to_stage2_fakeavceleb_mrdf5fold_fold1_encoder_full_audio_explicit/checkpoint-*`
- `/data/OneDay/OmniAV-Detect/data/swift_sft/fakeavceleb_mrdf5fold/fakeavceleb_mrdf5fold_fold1_binary_test_with_audio.jsonl`

输出：

- `/data/OneDay/OmniAV-Detect/outputs/batch_eval_binary/fakeavceleb_mrdf5fold_fold1_stage1_to_stage2_audio`

备注：

- 如果 full checkpoint 目录下存在 `checkpoint-*`，脚本会自动使用最新 checkpoint；否则直接使用该模型目录。
- 脚本强制关闭 `use_audio_in_video`，避免 `audios` 字段和视频内音频重复输入。

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

### MVAD: stage1 显式 audios baseline 训练

用途：使用 MVAD internal train JSONL 做 Qwen2.5-Omni stage1 LoRA baseline 训练。

```bash
bash mvad/train_stage1_MVAD.sh
```

输入：

- `/data/OneDay/OmniAV-Detect/data/swift_sft/mvad/mvad_binary_train_with_audio.jsonl`

输出：

- `/data/OneDay/OmniAV-Detect/outputs/stage1_qwen2_5_omni_mvad_binary_audio_explicit`

备注：

- 脚本显式设置 `USE_AUDIO_IN_VIDEO=False` 和 `use_audio_in_video=False`。
- 该训练结果只对应 MVAD public train-only internal split，不应表述为官方 test 结果。

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
