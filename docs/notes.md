# Notes

## 需求

清理评估代码，只保留两个批量入口脚本，并将内部评估入口全部下沉到 `src/omniav_detect/evaluation/`。

## 修改文件

- scripts/eval_batch_binary_qwen_omni_vllm.py
- src/omniav_detect/evaluation/__init__.py
- src/omniav_detect/evaluation/batch_runner.py
- src/omniav_detect/evaluation/parallel_runner.py
- src/omniav_detect/evaluation/worker_cli.py
- src/omniav_detect/evaluation/parallel_cli.py
- src/omniav_detect/evaluation/vllm_cli.py
- tests/test_eval_batch_binary_qwen_omni.py
- tests/test_eval_parallel_binary_qwen_omni.py
- tests/test_eval_binary_logits_qwen_omni.py
- tests/test_eval_binary_logits_qwen_omni_vllm.py
- docs/architecture.md
- docs/notes.md
- docs/logs/2026-05.md
- scripts/eval_parallel_binary_qwen_omni.py
- scripts/eval_binary_logits_qwen_omni.py
- scripts/eval_binary_logits_qwen_omni_vllm.py

## 修改内容

- 删除 3 个旧评估脚本，仅保留两个批量入口脚本。
- 批量评估后端改为通过 `python -m omniav_detect.evaluation.*` 调用内部模块入口。
- 并行 worker 改为通过内部 `worker_cli` 启动，不再依赖 `scripts/` 目录。
- 为子进程补齐 `PYTHONPATH=src`，避免依赖 editable install。
- 更新测试与架构文档，去掉已删除脚本名。

## 验证

```bash
python -B tests/test_eval_batch_binary_qwen_omni.py -v
python -B tests/test_eval_parallel_binary_qwen_omni.py -v
python -B tests/test_eval_binary_logits_qwen_omni.py -v
python -B tests/test_eval_binary_logits_qwen_omni_vllm.py -v
python -B scripts/eval_batch_binary_qwen_omni.py --dry_run --only stage1_qwen2_5_omni_fakeavceleb_binary__fakeavceleb_eval
python -B scripts/eval_batch_binary_qwen_omni_vllm.py --dry_run --only stage1_qwen2_5_omni_fakeavceleb_binary__fakeavceleb_eval
python -B -m py_compile src/omniav_detect/evaluation/batch_runner.py src/omniav_detect/evaluation/parallel_runner.py src/omniav_detect/evaluation/worker_cli.py src/omniav_detect/evaluation/parallel_cli.py src/omniav_detect/evaluation/vllm_cli.py scripts/eval_batch_binary_qwen_omni.py scripts/eval_batch_binary_qwen_omni_vllm.py
```

结果：通过

## Git

- branch: `refactor/remove-extra-eval-scripts`
- commit: `git commit -m "refactor: collapse evaluation to two batch entrypoints"`
