"""
本文件功能：
- 组织 Qwen2.5-Omni binary deepfake detector 的评估模块。

主要内容：
- binary_logits：单 checkpoint 的 Real/Fake token logits 评估。
- binary_logits_vllm：vLLM 后端的单 checkpoint logits 评估。
- model_runtime：Qwen2.5-Omni 模型加载、多模态输入处理和 logits forward。
- vllm_runtime：vLLM 推理、多模态输入处理和 logprob 解析。
- parallel_runner：按 GPU 切分 JSONL 并并发运行多个评估子进程。
- metrics：二分类指标和 sklearn 后备逻辑。
- outputs / visualization：评估结果写出和可视化文件生成。
- batch_runner：读取 YAML / JSON 配置并批量调度多个评估任务。

使用方式：
- 由 `scripts/eval_binary_logits_qwen_omni.py`、`scripts/eval_parallel_binary_qwen_omni.py`
  和 `scripts/eval_batch_binary_qwen_omni.py` 调用。
"""
