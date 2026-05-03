"""
本文件功能：
- 组织 Qwen2.5-Omni binary deepfake detection 的评估模块。

主要内容：
- binary_logits：Transformers 后端的单 worker 评估实现，供并行评估路径内部调用。
- binary_logits_vllm：vLLM 后端的单次评估主流程。
- model_runtime：Transformers 后端的模型加载、多模态输入处理和 logits forward。
- vllm_runtime：vLLM 推理、多模态输入处理和 logprob 解析。
- parallel_runner：按 GPU 切分任务并调度多个评估 worker。
- batch_runner：读取 YAML 配置并批量调度并行评估或 vLLM 评估。
- progress：评估进度条和 tqdm 退化封装。
- metrics：二分类指标和 sklearn 后备实现。
- outputs / visualization：评估结果写出和可视化生成。

使用方式：
- 用户主入口为 `scripts/eval_batch_binary_qwen_omni.py` 和
  `scripts/eval_batch_binary_qwen_omni_vllm.py`。
- 内部子进程入口统一位于 `parallel_cli.py`、`vllm_cli.py` 和 `worker_cli.py`。
"""
