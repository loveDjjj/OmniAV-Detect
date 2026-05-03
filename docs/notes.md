# Notes

## 需求

支持评估 `stage1 -> stage2` 得到的 full checkpoint，以及单独 stage2 full checkpoint；同时让 vLLM 路径不再强制要求 LoRA adapter。

## 修改文件

- src/omniav_detect/evaluation/binary_logits_vllm.py
- configs/eval/qwen_omni_binary_batch_eval.yaml
- configs/eval/qwen_omni_binary_batch_eval_vllm.yaml
- tests/test_eval_binary_logits_qwen_omni_vllm.py
- docs/notes.md
- docs/logs/2026-05.md

## 修改内容

- 将 vLLM 评估入口的 `adapter_path` 改为可选，支持直接加载 full checkpoint。
- 在并行评估和 vLLM 批量评估 YAML 中新增 `stage1_to_stage2_mavosdd_encoder_full__mavosdd_test_all` run。
- full checkpoint run 使用 `model_path=<checkpoint目录>`、`adapter_path` 留空的方式评估。

## 验证

```bash
python -B tests/test_eval_binary_logits_qwen_omni_vllm.py -v
python -B tests/test_eval_batch_binary_qwen_omni.py -v
python -B scripts/eval_batch_binary_qwen_omni.py --dry_run --only stage1_to_stage2_mavosdd_encoder_full__mavosdd_test_all
python -B scripts/eval_batch_binary_qwen_omni_vllm.py --dry_run --only stage1_to_stage2_mavosdd_encoder_full__mavosdd_test_all
python -B -m py_compile src/omniav_detect/evaluation/binary_logits_vllm.py
```

结果：通过

## Git

- branch: `feat/support-full-checkpoint-eval`
- commit: `git commit -m "feat: support full-checkpoint evaluation"`
