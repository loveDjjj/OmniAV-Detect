# OmniAV-Detect MVAD

## 项目简介

本分支是 MVAD 专用版本，只保留 MVAD 公开 train 数据的预处理、Qwen2.5-Omni / Qwen3-Omni baseline 训练脚本和对应文档。

当前目标：

- 解压 MVAD `train/**/*.zip`。
- 扫描 `real_real`、`real_fake`、`fake_real`、`fake_fake` 四类样本。
- 为每条视频配对显式音频路径。
- 按 group-aware 规则构建 internal train/val 划分。
- 生成 Qwen Omni / ms-swift 可用的显式 `audios` JSONL。
- 提供 Qwen2.5-Omni-7B 和 Qwen3-Omni-30B-A3B-Thinking 的 LoRA 训练脚本。

注意：MVAD 当前公开数据只有 train。本项目生成的是 internal validation baseline，不是官方 test 复现。

## 项目结构

```text
mvad/
  common.py                         # MVAD 路径、标签、JSONL 公共函数
  progress.py                       # tqdm 进度条封装
  unzip_archives.py                 # 7z / zipfile 解压
  pairing.py                        # 同目录音视频配对与内嵌音轨检测
  build_index_and_split.py          # 样本索引与 group-aware 划分
  build_av_jsonl.py                 # 显式 audios JSONL 构建
  prepare_mvad.py                   # 一体化预处理入口
  run_prepare_mvad.sh               # 服务器一键预处理命令
  train_stage1_MVAD.sh              # Qwen2.5-Omni-7B stage1 LoRA
  train_stage1_MVAD_Qwen3Omni30BThinking.sh
                                    # Qwen3-Omni-30B-A3B-Thinking stage1 LoRA 尝试脚本

docs/
  commands.md                       # 常用命令
  architecture.md                   # MVAD 流程说明
  notes.md                          # 当前状态摘要
  logs/2026-05.md                   # 变更记录

tests/
  test_mvad_prepare.py              # MVAD 预处理单元测试
```

## 数据规则

二分类标签：

- `real_real` -> `Real`
- `real_fake` / `fake_real` / `fake_fake` -> `Fake`

音频处理：

- 优先使用同目录同 stem 的音频文件，例如 `video_1.mp4` + `video_1.wav` / `video_1.flac`。
- 如果同目录没有音频，用 `ffprobe` 判断视频是否有内嵌音轨。
- 有内嵌音轨时抽到视频同目录同名 `.wav`。
- 同目录音频和内嵌音轨都不存在时，写入 `missing_audio_pairs.jsonl`，不进入训练 JSONL。

## 一键预处理

```bash
SOURCE_ROOT=/data/OneDay/MVAD bash mvad/run_prepare_mvad.sh
```

如果 zip 已经解压，推荐续跑：

```bash
SOURCE_ROOT=/data/OneDay/MVAD \
SKIP_UNZIP=true \
FFPROBE_WORKERS=16 \
FFMPEG_WORKERS=8 \
bash mvad/run_prepare_mvad.sh
```

默认输出：

- `/data/OneDay/OmniAV-Detect/data/mvad_unpacked`
- `/data/OneDay/OmniAV-Detect/data/mvad_processed`
- `/data/OneDay/OmniAV-Detect/data/swift_sft/mvad/mvad_binary_train_with_audio.jsonl`
- `/data/OneDay/OmniAV-Detect/data/swift_sft/mvad/mvad_binary_val_with_audio.jsonl`

## 训练

Qwen2.5-Omni-7B：

```bash
bash mvad/train_stage1_MVAD.sh
```

Qwen3-Omni-30B-A3B-Thinking：

```bash
bash mvad/train_stage1_MVAD_Qwen3Omni30BThinking.sh
```

Qwen3 脚本是尝试启动配置。30B 模型在普通 `swift sft` DDP LoRA 下，2x48GB 仍可能 OOM；如果无法启动，需要改用 Megatron-SWIFT / 张量并行或低比特训练方案。

## 验证

```bash
python -B tests/test_mvad_prepare.py -v
python -B -m py_compile mvad/common.py mvad/progress.py mvad/unzip_archives.py mvad/pairing.py mvad/build_index_and_split.py mvad/build_av_jsonl.py mvad/prepare_mvad.py
```
