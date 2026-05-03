# Notes

## 需求

将 `FPS_MAX_FRAMES=64`、`VIDEO_MAX_PIXELS=25088`、`MAX_PIXELS=501760` 只加到 MAVOS-DD 的评估里，并写进 YAML 配置。

## 修改文件

- src/omniav_detect/evaluation/batch_runner.py
- configs/eval/qwen_omni_binary_batch_eval.yaml
- configs/eval/qwen_omni_binary_batch_eval_vllm.yaml
- tests/test_eval_batch_binary_qwen_omni.py
- docs/notes.md
- docs/logs/2026-05.md

## 修改内容

- 为批量评估增加 run 级 `env` 配置，并在启动子进程时合并到环境变量。
- 在两个评估 YAML 中，只给 MAVOS-DD 的 run 增加 `FPS_MAX_FRAMES`、`VIDEO_MAX_PIXELS`、`MAX_PIXELS`。
- FakeAVCeleb 的 run 不带这些限制，保持原样。

## 验证

```bash
python -B tests/test_eval_batch_binary_qwen_omni.py -v
python -B tests/test_eval_binary_logits_qwen_omni_vllm.py -v
python -B scripts/eval_batch_binary_qwen_omni.py --dry_run --only stage1_to_stage2_mavosdd_encoder_full__mavosdd_test_all stage1_qwen2_5_omni_mavosdd_binary__mavosdd_test_all
python -B scripts/eval_batch_binary_qwen_omni_vllm.py --dry_run --only stage1_to_stage2_mavosdd_encoder_full__mavosdd_test_all stage1_qwen2_5_omni_mavosdd_binary__mavosdd_test_all
python -B -m py_compile src/omniav_detect/evaluation/batch_runner.py src/omniav_detect/evaluation/binary_logits_vllm.py scripts/eval_batch_binary_qwen_omni.py scripts/eval_batch_binary_qwen_omni_vllm.py
```

结果：通过

## Git

- branch: `feat/mavosdd-eval-env-limits`
- commit: `git commit -m "feat: add MAVOS-DD evaluation env limits"`
