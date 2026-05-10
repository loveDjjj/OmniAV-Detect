# Notes

## 当前定位

本分支已裁剪为 MVAD-only 项目，只保留 MVAD 数据预处理、显式 audio-video JSONL 生成、Qwen Omni 训练脚本和对应测试/文档。

## 保留内容

- `mvad/`：MVAD 预处理与训练入口。
- `docs/`：MVAD 架构、命令、记录和参考文档。
- `tests/test_mvad_prepare.py`：MVAD 预处理单测。
- `README.md`、`AGENTS.md`、`.gitignore`、`requirements.txt`。

## 移除内容

- FakeAVCeleb / MAVOS-DD 数据准备配置和脚本。
- Qwen binary eval 批量评估代码。
- 旧 reports 和非 MVAD 训练脚本。
- 原 `src/omniav_detect` 包。

## 当前 MVAD 功能

- 默认用 `7z x` 解压 zip。
- 跳过 macOS `__MACOSX` 和 `._*` 资源叉文件。
- 支持同目录同 stem 音视频配对。
- 同目录无音频时用并发 `ffprobe` 检测内嵌音轨。
- 内嵌音轨用并发 `ffmpeg` 抽到同目录 `.wav`。
- 无音频样本写入 `missing_audio_pairs.jsonl`，不进入训练 JSONL。
- 按 `group_id` 做 train/val 划分，避免同源组泄漏。
- 输出 Qwen Omni / ms-swift 显式 `audios` JSONL。

## 训练入口

- `mvad/train_stage1_MVAD.sh`：Qwen2.5-Omni-7B LoRA baseline。
- `mvad/train_stage1_MVAD_Qwen3Omni30BThinking.sh`：Qwen3-Omni-30B-A3B-Thinking LoRA 尝试。

两个训练脚本都关闭 `use_audio_in_video`，避免 JSONL `audios` 和视频内音频重复输入。

## 验证

```powershell
python -B tests\test_mvad_prepare.py -v
python -B -m py_compile mvad\common.py mvad\progress.py mvad\unzip_archives.py mvad\pairing.py mvad\build_index_and_split.py mvad\build_av_jsonl.py mvad\prepare_mvad.py
```
