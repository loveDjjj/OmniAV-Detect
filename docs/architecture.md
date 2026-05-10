# Architecture

## 定位

本分支只保留 MVAD 数据准备与 Qwen Omni 训练入口。FakeAVCeleb、MAVOS-DD、批量评估和报告汇总代码已经从该分支移除。

## 模块划分

- `mvad/common.py`：MVAD 文件扫描、标签映射、group_id 构造、JSON/JSONL 写出。
- `mvad/progress.py`：tqdm 进度条封装，缺少 tqdm 时退化为空实现。
- `mvad/unzip_archives.py`：递归查找并解压 zip，默认使用 `7z x`。
- `mvad/pairing.py`：同目录音视频配对，必要时并发调用 `ffprobe` 检查内嵌音轨。
- `mvad/build_index_and_split.py`：构建样本索引、缺失音频报告和 group-aware train/val 划分。
- `mvad/build_av_jsonl.py`：根据 index 写出 Qwen Omni 显式 `audios` JSONL，并并发调用 `ffmpeg` 抽取内嵌音频。
- `mvad/prepare_mvad.py`：一体化 CLI 入口。
- `mvad/run_prepare_mvad.sh`：服务器预处理命令封装。
- `mvad/train_stage1_MVAD.sh`：Qwen2.5-Omni-7B LoRA 训练。
- `mvad/train_stage1_MVAD_Qwen3Omni30BThinking.sh`：Qwen3-Omni-30B-A3B-Thinking LoRA 训练尝试。

## 数据流

1. `run_prepare_mvad.sh` 调用 `python -m mvad.prepare_mvad`。
2. 如未设置 `SKIP_UNZIP=true`，先将 `SOURCE_ROOT/train/**/*.zip` 解压到 `UNPACK_ROOT`。
3. 扫描解压目录下真实视频，跳过 `__MACOSX` 和 `._*` 资源叉文件。
4. 从路径推断四类模态：`real_real`、`real_fake`、`fake_real`、`fake_fake`。
5. 优先按同目录同 stem 配对音频文件。
6. 找不到分离音频时，用 `ffprobe` 判断视频内嵌音轨；有音轨则登记同目录同名 `.wav` 作为抽取目标。
7. 没有任何音频来源的样本写入 `missing_audio_pairs.jsonl`，不进入训练 JSONL。
8. 按 `group_id` 执行 train/val 划分，避免同源组跨 split。
9. 写出 `mvad_train_index.jsonl`、`mvad_val_index.jsonl`、`split_stats.json`。
10. 根据 index 生成 `mvad_binary_train_with_audio.jsonl` 和 `mvad_binary_val_with_audio.jsonl`。

## 训练约束

MVAD JSONL 已经显式包含 `audios`，训练时必须关闭视频内重复抽音频：

```bash
export USE_AUDIO_IN_VIDEO=False
export use_audio_in_video=False
```

训练脚本默认不跑验证集：

```bash
--split_dataset_ratio 0
--eval_strategy no
```

## 输出文件

默认服务器输出：

- `/data/OneDay/OmniAV-Detect/data/mvad_unpacked`
- `/data/OneDay/OmniAV-Detect/data/mvad_processed/mvad_all_index.jsonl`
- `/data/OneDay/OmniAV-Detect/data/mvad_processed/mvad_train_index.jsonl`
- `/data/OneDay/OmniAV-Detect/data/mvad_processed/mvad_val_index.jsonl`
- `/data/OneDay/OmniAV-Detect/data/mvad_processed/missing_audio_pairs.jsonl`
- `/data/OneDay/OmniAV-Detect/data/mvad_processed/split_stats.json`
- `/data/OneDay/OmniAV-Detect/data/swift_sft/mvad/mvad_binary_train_with_audio.jsonl`
- `/data/OneDay/OmniAV-Detect/data/swift_sft/mvad/mvad_binary_val_with_audio.jsonl`
