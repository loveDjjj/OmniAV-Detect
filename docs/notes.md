# Notes

## 需求

新增 stage1 -> stage2 训练脚本：先 merge stage1 LoRA，再按现有 stage2 思路继续训练，并补总控脚本串行执行两个数据集。

## 修改文件

- docs/notes.md
- docs/commands.md
- docs/architecture.md
- docs/logs/2026-05.md
- train_stage1_to_stage2_FakeAVCeleb.sh
- train_stage1_to_stage2_MAVOS-DD.sh
- train_all_stage1_stage2.sh

## 修改内容

- 新增 FakeAVCeleb 和 MAVOS-DD 的 `train_stage1_to_stage2_*.sh`。
- 每个脚本会先自动选择 `stage1` 输出目录下最新的 `checkpoint-*`，再用 `swift export --merge_lora true` 将 LoRA merge 到基模。
- merge 完成后，继续沿用当前 `stage2` 的 `full + freeze_llm` 训练思路。
- 新增 `train_all_stage1_stage2.sh`，顺序执行两个数据集脚本并将日志写入 `logs/`。

## 验证

```bash
Get-Content train_stage1_to_stage2_FakeAVCeleb.sh
Get-Content train_stage1_to_stage2_MAVOS-DD.sh
Get-Content train_all_stage1_stage2.sh
```

结果：人工检查通过

## Git

- branch: `feat/train-stage1-to-stage2-scripts`
- commit: `git commit -m "feat: add stage1 to stage2 training scripts"`
