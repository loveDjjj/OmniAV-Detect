# Notes

## 需求

新增独立 `mvad/` 目录，实现 MVAD 公开 train 数据的 zip 解压、音频抽取、group-aware train/val 划分、Qwen2.5-Omni JSONL 生成，以及 stage1 baseline 训练 bash 命令。

## 修改文件

- mvad/
- tests/test_mvad_prepare.py
- README.md
- docs/architecture.md
- docs/commands.md
- docs/notes.md
- docs/logs/2026-05.md

## 修改内容

- 新增 MVAD 独立预处理包，支持解压 zip、扫描视频、映射四类模态标签、构造 group_id、按 group-aware 规则划分 internal train/val。
- 新增显式音频 JSONL 生成逻辑，调用 ffmpeg 抽取 16 kHz mono wav，并生成 `mvad_binary_train_with_audio.jsonl` / `mvad_binary_val_with_audio.jsonl`。
- 新增 `mvad/run_prepare_mvad.sh` 和 `mvad/train_stage1_MVAD.sh`，用于一键预处理和 stage1 LoRA baseline 训练。
- 更新 README、架构和命令文档，说明 MVAD 当前只能作为 public train-only internal validation baseline。

## 验证

```powershell
python -B tests\test_mvad_prepare.py -v
python -B -m py_compile mvad\common.py mvad\unzip_archives.py mvad\build_index_and_split.py mvad\build_av_jsonl.py mvad\prepare_mvad.py
bash -n mvad/run_prepare_mvad.sh
bash -n mvad/train_stage1_MVAD.sh
```

结果：通过。补充运行了临时 zip fixture 的 `python -m mvad.prepare_mvad --dry_run`，已生成 train/val JSONL、index 和 split stats。`tests/test_project_structure.py` 当前失败在已有 `src/omniav_detect/data/fakeavceleb.py`、`src/omniav_detect/evaluation/batch_runner.py`、`src/omniav_detect/evaluation/parallel_runner.py` 超过 500 行，非本次新增 `mvad/` 代码导致。

## Git

- branch: `feat-mvad-preprocess`
- commit: `feat: add mvad preprocessing pipeline`
