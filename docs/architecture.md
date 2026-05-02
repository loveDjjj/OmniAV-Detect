# Architecture

## 总体流程

项目流程分三段：本地 FakeAVCeleb / MAVOS-DD 视频转换为 ms-swift SFT JSONL；外部 `swift sft` 训练 Qwen2.5-Omni LoRA；仓库评估脚本读取 JSONL，并基于 Qwen2.5-Omni 基模或 LoRA checkpoint 的 `Real` / `Fake` token logits 计算二分类指标。

## 模块划分

- 配置加载：`src/omniav_detect/config.py` 统一读取 YAML / JSON。
- 数据准备调度：`src/omniav_detect/data/prepare_runner.py` 读取 `configs/data/swift_av_sft.yaml`，按 `--dataset` 只处理一个数据集。
- 数据公共逻辑：`src/omniav_detect/data/common.py` 负责视频扫描、路径规范化、SFT record 构造和统计写出。
- FakeAVCeleb：`src/omniav_detect/data/fakeavceleb.py` 负责四类目录扫描、metadata merge、`overall_label + modality_type` 分层切分。
- MAVOS-DD：`src/omniav_detect/data/mavosdd.py` 负责 `datasets.load_from_disk`、视频路径解析和 open-set split 归类。
- 单模型评估入口：`src/omniav_detect/evaluation/binary_logits.py` 负责任务编排、失败样本隔离和增量保存。
- 模型运行：`src/omniav_detect/evaluation/model_runtime.py` 加载 Qwen2.5-Omni 基座和可选 LoRA adapter，处理多模态输入并执行 logits forward。
- vLLM 评估入口：`src/omniav_detect/evaluation/binary_logits_vllm.py` 负责 vLLM 后端的评估主流程。
- vLLM 运行时：`src/omniav_detect/evaluation/vllm_runtime.py` 负责 vLLM 引擎加载、多模态输入和 logprob 解析。
- 并行评估：`src/omniav_detect/evaluation/parallel_runner.py` 负责把 JSONL 按 GPU 切成 shard，启动多个单模型评估子进程，并合并预测重算指标。
- 指标计算：`src/omniav_detect/evaluation/metrics.py` 计算 Accuracy、AUC、AP/mAP、Confusion Matrix、Fake recall 和 Real recall。
- 评估输出：`src/omniav_detect/evaluation/outputs.py` 写出预测、坏样本、指标和可视化；`visualization.py` 生成 CSV/HTML，可选生成 PNG。
- 批量评估：`src/omniav_detect/evaluation/batch_runner.py` 读取 `configs/eval/qwen_omni_binary_batch_eval.yaml`，逐个子进程调用单模型评估。

当前未发现仓库内自定义模型结构、loss 或训练循环；训练由外部 `swift sft` 命令负责。

当前仓库还包含一组服务器训练脚本：`train_stage1_*.sh`、`train_stage2_*.sh` 和 `train_stage1_to_stage2_*.sh`。其中 `stage1_to_stage2` 脚本会先 merge stage1 LoRA，再用 merge 后模型作为 stage2 `full + freeze_llm` 的初始化权重。

## 关键调用关系

- `scripts/prepare_swift_av_sft.py` → `omniav_detect.data.prepare_runner.main`
- `prepare_runner.prepare_dataset` → `fakeavceleb.build_fakeavceleb_samples` 或 `mavosdd.build_mavosdd_samples`
- `scripts/eval_binary_logits_qwen_omni.py` → `omniav_detect.evaluation.binary_logits.main`
- `scripts/eval_parallel_binary_qwen_omni.py` → `omniav_detect.evaluation.parallel_runner.main`
- `scripts/eval_binary_logits_qwen_omni_vllm.py` → `omniav_detect.evaluation.binary_logits_vllm.main`
- `scripts/eval_batch_binary_qwen_omni.py` → `omniav_detect.evaluation.batch_runner.main`
- `batch_runner.build_eval_command` → 子进程执行 `scripts/eval_binary_logits_qwen_omni.py`

## 数据流

数据准备：

1. `prepare_runner` 读取 YAML，合并默认值、数据集配置和命令行覆盖参数。
2. 数据集 builder 扫描或读取 metadata，并跳过缺失、空文件、非法扩展名文件。
3. `common.make_binary_record` / `make_structured_record` 构造 ms-swift JSONL 记录。
4. `common.write_output_jsonl` 写训练文件，`common.write_stats` 写扫描、统计、缺失异常和预览文件。

Transformers 评估：

1. `binary_logits` 读取 ms-swift JSONL，取 `videos[0]` 和 `meta.overall_label`。
2. 构造与训练一致的 Qwen2.5-Omni conversation。
3. 优先使用 `qwen_omni_utils.process_mm_info` 处理视频/音频输入。
4. 加载 Qwen2.5-Omni 基座和 PEFT LoRA adapter，定位 logits forward 模块。
5. 取最后位置 logits，在 `Real` / `Fake` token 上 softmax，生成预测和指标。
6. 写出 `metrics.json` 后，同步生成 `visualizations/confusion_matrix.csv`、`score_distribution.csv` 和 `summary.html`。

vLLM 评估：

1. `binary_logits_vllm` 读取同一份 ms-swift JSONL，并构造同一套 system / user prompt。
2. 默认 `--mm_format omni_av` 使用 `qwen_omni_utils.process_mm_info` 构造 audio+video 输入；只有显式设置 `--mm_format video --no_use_audio_in_video` 时才走 video-only 路径。
3. vLLM 输出端优先请求 `Real` / `Fake` token 的 logprob；如果当前 vLLM 版本不支持 `logprob_token_ids`，则用 `--logprobs -1` 请求完整分布后再提取二分类概率。

并行评估：

1. `parallel_runner` 先把输入 JSONL 按 round-robin 切为多个 shard，避免原始文件按类别排序时某个 worker 只拿到单一类别。
2. 每个 worker 通过独立子进程运行 `scripts/eval_binary_logits_qwen_omni.py`，并只暴露一张 GPU，例如 `CUDA_VISIBLE_DEVICES=0`。
3. 每个 worker 独立加载基座模型和 LoRA adapter；这属于数据并行评估，不共享模型进程。
4. 所有 worker 完成后，合并 `predictions.jsonl` 和 `bad_samples.jsonl`，再用 `metrics.compute_metrics` 重新计算整体 Acc、AUC、AP/mAP、Confusion Matrix、Fake recall 和 Real recall。

## 配置流

- `configs/data/swift_av_sft.yaml` 保存数据集 root、output_dir、seed、preview 数量和 FakeAVCeleb 切分参数。
- `scripts/prepare_swift_av_sft.py --dataset <name>` 选择单个数据集；`--root`、`--output_dir`、`--mode` 等参数只覆盖本次运行。
- `configs/eval/qwen_omni_binary_batch_eval.yaml` 保存默认模型路径、输出根目录、批量 run 列表和默认 batch/fps/token 参数。
- `eval_batch_binary_qwen_omni.py` 支持运行时覆盖 `batch_size`、`max_samples`、`fps`、`save_every`、`output_root`。
- `eval_parallel_binary_qwen_omni.py` 不读取 YAML，主要通过命令行传入 `--gpus`、`--num_workers`、`--batch_size` 和 `--fps`。

## 输出文件

数据准备输出：

- `data/swift_sft/fakeavceleb/*.jsonl`
- `data/swift_sft/mavosdd/*.jsonl`
- `dataset_scan_summary.json`
- `dataset_stats.json`
- `missing_or_invalid_files.csv`
- `preview_samples.json`

评估输出：

- `predictions.jsonl`
- `bad_samples.jsonl`
- `metrics.json`
- `visualizations/confusion_matrix.csv`
- `visualizations/score_distribution.csv`
- `visualizations/summary.html`
- `visualizations/*.png`（仅 matplotlib 可用时）
- `batch_eval_summary.json`
- `batch_eval_summary.csv`
- `parallel_manifest.json`
- `parallel_status.json`
- `workers/worker_*/`

训练输出：

- `outputs/` 下的 LoRA checkpoint、日志和中间结果，具体由 ms-swift 控制。

## 待确认

- conda 环境名待确认。
- Python 版本待确认。
- 服务器上 Qwen2.5-Omni、torch、transformers、peft、ms-swift 的精确版本待确认。
