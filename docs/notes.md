# Notes

## 需求

修复 vLLM 后端的多模态视频输入构造，避免把视频路径字符串直接传给 vLLM。

## 修改文件

- requirements.txt
- docs/notes.md
- docs/commands.md
- docs/logs/2026-05.md
- src/omniav_detect/evaluation/binary_logits_vllm.py
- src/omniav_detect/evaluation/vllm_runtime.py
- tests/test_eval_binary_logits_qwen_omni_vllm.py

## 修改内容

- vLLM 后端使用 `qwen-vl-utils.process_vision_info` 生成视频输入，避免传入字符串路径。
- `mm_format` 增加 `qwen_vl/qwen_vl_utils/video_path` 等选项，并默认使用 vLLM 兼容的视频输入。
- 增加 `qwen-vl-utils` 依赖说明与相关测试覆盖。

## 验证

```bash
D:/anaconda/envs/oneday/python.exe -B -m unittest discover -s tests -v
```

结果：通过
