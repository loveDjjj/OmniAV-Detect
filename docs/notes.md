# Notes

## 需求

为 FakeAVCeleb MRDF 5-fold fold1 的显式 `audios` 数据集补齐测评适配，并给出 stage1 与 stage1->stage2 的简单 shell 测评命令。

## 修改文件

- src/omniav_detect/evaluation/data_io.py
- src/omniav_detect/evaluation/model_runtime.py
- src/omniav_detect/evaluation/vllm_runtime.py
- configs/eval/qwen_omni_binary_batch_eval.yaml
- eval_fakeavceleb_mrdf5fold_fold1_stage1_audio.sh
- eval_fakeavceleb_mrdf5fold_fold1_stage1_to_stage2_audio.sh
- tests/test_eval_binary_logits_qwen_omni.py
- tests/test_eval_batch_binary_qwen_omni.py
- docs/commands.md
- docs/notes.md
- docs/logs/2026-05.md

## 修改内容

- 评估 JSONL 读取逻辑保留 `audios` 字段，并规范化为 `audio_paths`。
- Transformers 与 vLLM evaluation conversation 均支持显式 audio item。
- 新增 fold1 stage1 / stage1->stage2 两个直接运行的评估脚本，均强制 `--no_use_audio_in_video`。
- 并行评估 YAML 增加两个 MRDF fold1 显式音频 run，便于后续统一入口管理。

## 验证

```bash
python -B tests/test_eval_binary_logits_qwen_omni.py -v
python -B tests/test_eval_batch_binary_qwen_omni.py -v
python -B -m py_compile src/omniav_detect/evaluation/data_io.py src/omniav_detect/evaluation/model_runtime.py src/omniav_detect/evaluation/vllm_runtime.py
```

结果：待运行

## Git

- branch: `feat/fakeavceleb-mrdf-eval-audio`
- commit: `git commit -m "feat: add fakeavceleb mrdf audio evaluation"`
