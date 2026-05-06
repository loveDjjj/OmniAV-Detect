# Notes

## 需求

为 FakeAVCeleb 新增一种参考 MRDF 的 subject-independent 5 折划分方式，并补一个从视频 JSONL 抽取音频、生成带 `audios` 字段数据集的脚本，供 Qwen 多模态训练使用。

## 修改文件

- configs/data/swift_av_sft.yaml
- README.md
- docs/notes.md
- docs/commands.md
- docs/architecture.md
- docs/logs/2026-05.md
- .gitignore
- src/omniav_detect/data/common.py
- src/omniav_detect/data/fakeavceleb.py
- src/omniav_detect/data/prepare_runner.py
- scripts/extract_audio_and_build_av_jsonl.py
- train_stage1_FakeAVCeleb_MRDF5Fold_Audio.sh
- train_stage1_to_stage2_FakeAVCeleb_MRDF5Fold_Audio.sh
- tests/test_prepare_swift_av_sft.py
- tests/test_extract_audio_and_build_av_jsonl.py

## 修改内容

- 为 FakeAVCeleb 增加 `split_protocol`，支持默认 `random_stratified` 和 `mrdf_5fold` 两种 protocol。
- 新增 `fakeavceleb_mrdf5fold` 配置项，默认读取 `/data/.../data_fakeavceleb/train_*.txt` 与 `test_*.txt`。
- 在 FakeAVCeleb 样本 meta 中补充 `subject_id`，用于和 MRDF 5 折文件做 subject-independent 匹配。
- 新增音频抽取脚本：从已有视频 JSONL 批量抽取 wav，并生成带 `audios` 字段的新 JSONL。
- 扩展公共 record 构造逻辑：样本带音频路径时，自动写入 `audios` 并在 user prompt 中显式包含 `<audio>`。
- 新增 FakeAVCeleb MRDF 5 折 + 显式 audios 的 stage1 和 stage1->stage2 训练脚本，并强制关闭 `use_audio_in_video`，避免音频重复输入。

## 验证

```bash
python -B -m unittest tests.test_prepare_swift_av_sft tests.test_extract_audio_and_build_av_jsonl -v
python -B -m py_compile scripts/extract_audio_and_build_av_jsonl.py src/omniav_detect/data/common.py src/omniav_detect/data/fakeavceleb.py src/omniav_detect/data/prepare_runner.py
python -B scripts/prepare_swift_av_sft.py --dataset fakeavceleb_mrdf5fold --dry_run --num_preview 2
Get-Content train_stage1_FakeAVCeleb_MRDF5Fold_Audio.sh
Get-Content train_stage1_to_stage2_FakeAVCeleb_MRDF5Fold_Audio.sh
```

结果：待运行

## Git

- branch: `feat/fakeavceleb-mrdf5fold-audio`
- commit: `git commit -m "feat: add fakeavceleb mrdf 5-fold preparation"`
