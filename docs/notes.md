# Notes

## 需求

修复 vLLM 后端未返回 logprobs 导致无法读取 Real/Fake 概率的问题。

## 修改文件

- docs/notes.md
- docs/commands.md
- docs/logs/2026-05.md
- src/omniav_detect/evaluation/binary_logits_vllm.py
- src/omniav_detect/evaluation/vllm_runtime.py
- tests/test_eval_binary_logits_qwen_omni_vllm.py

## 修改内容

- vLLM 评估强制请求 logprobs，并新增 `--logprobs` 参数（默认 -1）。
- logprobs 缺失时提示用户设置 `--logprobs` 以返回 Real/Fake token 概率。
- 更新命令与测试覆盖 logprobs 默认值。

## 验证

```bash
D:/anaconda/envs/oneday/python.exe -B -m unittest discover -s tests -v
```

结果：通过
