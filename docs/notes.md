# Notes

## 需求

新增一个单独运行 FakeAVCeleb MRDF 5-fold Stage 2 only 的训练脚本，参考 `train_stage2_FakeAVCeleb.sh`，但适配显式 `audios` 数据集。

## 修改文件

- train_stage2_FakeAVCeleb_MRDF5Fold_Audio.sh
- docs/commands.md
- docs/notes.md
- docs/logs/2026-05.md

## 修改内容

- 新增 `train_stage2_FakeAVCeleb_MRDF5Fold_Audio.sh`，默认 `FOLD_ID=1`，可切换 1..5 折。
- 训练策略沿用 Stage 2 only：`tuner_type=full`、`freeze_llm=true`、`freeze_vit=false`、`freeze_aligner=false`。
- 数据集路径指向 `fakeavceleb_mrdf5fold_fold${FOLD_ID}_binary_train_with_audio.jsonl`。
- 强制关闭 `USE_AUDIO_IN_VIDEO` / `use_audio_in_video`，避免显式 `audios` 与视频内音频重复输入。

## 验证

```bash
Get-Content train_stage2_FakeAVCeleb_MRDF5Fold_Audio.sh
bash -n train_stage2_FakeAVCeleb_MRDF5Fold_Audio.sh
```

结果：部分通过（`bash -n` 在当前 Windows/WSL 环境输出 E_ACCESSDENIED，不能作为可靠结果；脚本文本与 docs 命令条目检查通过）

## Git

- branch: `feat/fakeavceleb-mrdf-stage2-audio`
- commit: `git commit -m "feat: add fakeavceleb mrdf stage2 training script"`
