# Notes

## 需求

新增 vLLM 后端的 Qwen2.5-Omni binary logits 评估入口，保留现有 Transformers 路径不变。

## 修改文件

- README.md
- requirements.txt
- docs/notes.md
- docs/commands.md
- docs/architecture.md
- docs/logs/2026-05.md
- src/omniav_detect/evaluation/__init__.py
- src/omniav_detect/evaluation/binary_logits_vllm.py
- src/omniav_detect/evaluation/vllm_runtime.py
- scripts/eval_binary_logits_qwen_omni_vllm.py
- tests/test_eval_binary_logits_qwen_omni_vllm.py

## 修改内容

- 新增 vLLM 后端评估主流程与运行时模块，保持输出与指标契约不变。
- 新增 vLLM 薄入口脚本，并提供多模态输入格式参数。
- 文档补充 vLLM 评估命令、架构说明和依赖提示。

## 验证

```bash
D:/anaconda/envs/oneday/python.exe -B -m unittest discover -s tests -v
```

结果：通过
