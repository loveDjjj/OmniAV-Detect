# MVAD 预处理与 Qwen2.5-Omni baseline

本目录提供独立的 MVAD 数据准备流程，用于把当前公开的 `train` 数据整理为 Qwen2.5-Omni SFT JSONL。

## 定位

当前公开 MVAD 只有 `train` 数据，因此这里生成的是 internal train/val baseline，不是论文官方 test 复现。

二分类标签规则：

- `real_real` -> `Real`
- `real_fake` / `fake_real` / `fake_fake` -> `Fake`

## 一键预处理

```bash
SOURCE_ROOT=/data/MVAD bash mvad/run_prepare_mvad.sh
```

默认使用 `7z x` 解压 zip，并在解压、JSONL 构建和必要的抽音频阶段显示进度条。可通过 `EXTRACTOR=7z`、`FFMPEG=ffmpeg`、`FFPROBE=ffprobe` 覆盖命令。

音频处理规则：

- 优先按同目录同 stem 的视频和音频文件配对，例如 `video_jimeng_1.mp4` + `video_jimeng_1.wav`。
- 同目录没有音频时，用 `ffprobe` 检测视频是否有内嵌音轨；有音轨则抽到视频同目录同名 `.wav`。
- 同目录音频和内嵌音轨都不存在的样本写入 `mvad_processed/missing_audio_pairs.jsonl`，不进入训练 JSONL。
- 不给无音轨视频生成静音音频。

默认输出：

- `/data/OneDay/OmniAV-Detect/data/mvad_unpacked`
- `/data/OneDay/OmniAV-Detect/data/mvad_processed`
- `/data/OneDay/OmniAV-Detect/data/audio_cache/mvad`
- `/data/OneDay/OmniAV-Detect/data/swift_sft/mvad/mvad_binary_train_with_audio.jsonl`
- `/data/OneDay/OmniAV-Detect/data/swift_sft/mvad/mvad_binary_val_with_audio.jsonl`
- `/data/OneDay/OmniAV-Detect/data/mvad_processed/missing_audio_pairs.jsonl`

常用覆盖：

```bash
SOURCE_ROOT=/data/MVAD \
VAL_RATIO=0.1 \
SEED=42 \
bash mvad/run_prepare_mvad.sh
```

如果 zip 已经解压，只想重建划分和 JSONL：

```bash
SKIP_UNZIP=true SKIP_AUDIO=true bash mvad/run_prepare_mvad.sh
```

## Stage1 baseline 训练

```bash
bash mvad/train_stage1_MVAD.sh
```

训练脚本默认使用：

- 数据集：`/data/OneDay/OmniAV-Detect/data/swift_sft/mvad/mvad_binary_train_with_audio.jsonl`
- 输出：`/data/OneDay/OmniAV-Detect/outputs/stage1_qwen2_5_omni_mvad_binary_audio_explicit`

由于 JSONL 已经写入 `audios` 字段，训练脚本会显式关闭 `USE_AUDIO_IN_VIDEO` / `use_audio_in_video`。
