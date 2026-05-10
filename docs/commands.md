# Commands

## 安装依赖

```bash
pip install -r requirements.txt
```

服务器还需要系统命令：

```bash
which 7z
which ffmpeg
which ffprobe
```

## MVAD 预处理

首次完整预处理：

```bash
SOURCE_ROOT=/data/OneDay/MVAD bash mvad/run_prepare_mvad.sh
```

解压已完成后的推荐续跑：

```bash
SOURCE_ROOT=/data/OneDay/MVAD \
SKIP_UNZIP=true \
FFPROBE_WORKERS=16 \
FFMPEG_WORKERS=8 \
bash mvad/run_prepare_mvad.sh
```

常用环境变量：

- `SOURCE_ROOT`：MVAD 原始下载目录，默认 `/data/MVAD`。
- `UNPACK_ROOT`：解压输出目录，默认 `/data/OneDay/OmniAV-Detect/data/mvad_unpacked`。
- `WORK_ROOT`：index/stat 输出目录，默认 `/data/OneDay/OmniAV-Detect/data/mvad_processed`。
- `JSONL_ROOT`：Qwen JSONL 输出目录，默认 `/data/OneDay/OmniAV-Detect/data/swift_sft/mvad`。
- `SKIP_UNZIP=true`：跳过解压，直接从现有 `UNPACK_ROOT` 构建 index 和 JSONL。
- `SKIP_AUDIO=true`：跳过 ffmpeg 抽取，仅重建 JSONL。
- `OVERWRITE=true`：覆盖已有解压或音频输出。
- `SKIP_BAD_ARCHIVES=true`：跳过坏 zip 并记录到 manifest。
- `FFPROBE_WORKERS`：并发检测内嵌音轨数，默认 8。
- `FFMPEG_WORKERS`：并发抽取内嵌音频数，默认 4。

查看统计：

```bash
cat /data/OneDay/OmniAV-Detect/data/mvad_processed/split_stats.json
wc -l /data/OneDay/OmniAV-Detect/data/mvad_processed/missing_audio_pairs.jsonl
```

确认 JSONL 中视频和音频文件真实存在：

```bash
python - <<'PY'
import json
from pathlib import Path

files = [
    '/data/OneDay/OmniAV-Detect/data/swift_sft/mvad/mvad_binary_train_with_audio.jsonl',
    '/data/OneDay/OmniAV-Detect/data/swift_sft/mvad/mvad_binary_val_with_audio.jsonl',
]
for fp in files:
    total = missing_video = missing_audio = 0
    for line in Path(fp).open(encoding='utf-8'):
        if not line.strip():
            continue
        total += 1
        row = json.loads(line)
        missing_video += not Path(row['videos'][0]).exists()
        missing_audio += not Path(row['audios'][0]).exists()
    print(fp, 'total=', total, 'missing_video=', missing_video, 'missing_audio=', missing_audio)
PY
```

## Qwen2.5-Omni 训练

```bash
bash mvad/train_stage1_MVAD.sh
```

默认模型：

```bash
/data/OneDay/models/qwen/Qwen2.5-Omni-7B
```

默认输出：

```bash
/data/OneDay/OmniAV-Detect/outputs/stage1_qwen2_5_omni_mvad_binary_audio_explicit
```

脚本默认：

- 训练 2 epoch。
- 不跑 val：`--split_dataset_ratio 0`、`--eval_strategy no`。
- 关闭视频内重复抽音频。
- `FPS=1.0`、`FPS_MAX_FRAMES=32`、`MAX_LENGTH=4096`。

如果仍出现 `Failed to retrieve the dataset`，先尝试：

```bash
MAX_LENGTH=8192 FPS_MAX_FRAMES=16 bash mvad/train_stage1_MVAD.sh
```

## Qwen3-Omni-30B-A3B-Thinking 训练尝试

```bash
bash mvad/train_stage1_MVAD_Qwen3Omni30BThinking.sh
```

默认模型：

```bash
/data/OneDay/models/Qwen3-Omni-30B-A3B-Thinking
```

默认输出：

```bash
/data/OneDay/OmniAV-Detect/outputs/stage1_qwen3_omni_30b_a3b_thinking_mvad_binary_audio_explicit
```

注意：Qwen3-Omni-30B 在普通 `swift sft` DDP LoRA 下，2x48GB 仍可能 OOM。脚本默认已经使用较保守设置：

- `FPS=0.5`
- `FPS_MAX_FRAMES=16`
- `PER_DEVICE_TRAIN_BATCH_SIZE=1`
- `GRADIENT_ACCUMULATION_STEPS=32`

如果仍 OOM，需要考虑 Megatron-SWIFT / tensor parallel 或低比特训练。

## 本地验证

```bash
python -B tests/test_mvad_prepare.py -v
python -B -m py_compile mvad/common.py mvad/progress.py mvad/unzip_archives.py mvad/pairing.py mvad/build_index_and_split.py mvad/build_av_jsonl.py mvad/prepare_mvad.py
```
