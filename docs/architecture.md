# Architecture

## 总体流程

项目当前包含三条主线：

1. 将 FakeAVCeleb / MAVOS-DD 本地视频整理为 ms-swift / Qwen2.5-Omni 可直接使用的 SFT JSONL；MVAD 通过根目录 `mvad/` 的独立管线整理公开 train 数据。
2. 通过外部 `swift sft` 命令训练 Qwen2.5-Omni 基模或 LoRA。
3. 读取 SFT JSONL，用 Qwen2.5-Omni 的 `Real` / `Fake` token logits 做二分类评估。

评估对外只保留两条入口：

- `scripts/eval_batch_binary_qwen_omni.py`：并行评估后端
- `scripts/eval_batch_binary_qwen_omni_vllm.py`：vLLM 后端

## 模块划分

- 配置加载：`src/omniav_detect/config.py`
- 数据准备调度：`src/omniav_detect/data/prepare_runner.py`
- 数据公共逻辑：`src/omniav_detect/data/common.py`
- FakeAVCeleb 处理：`src/omniav_detect/data/fakeavceleb.py`
- MAVOS-DD 处理：`src/omniav_detect/data/mavosdd.py`
- 音频抽取与 AV JSONL 增强：`scripts/extract_audio_and_build_av_jsonl.py`
- MVAD 独立预处理：`mvad/common.py`、`unzip_archives.py`、`build_index_and_split.py`、`build_av_jsonl.py`、`prepare_mvad.py`
- 批量评估调度：`src/omniav_detect/evaluation/batch_runner.py`
- 并行评估调度：`src/omniav_detect/evaluation/parallel_runner.py`
- Transformers 单 worker 评估：`src/omniav_detect/evaluation/binary_logits.py`
- vLLM 评估：`src/omniav_detect/evaluation/binary_logits_vllm.py`
- Transformers 运行时：`src/omniav_detect/evaluation/model_runtime.py`
- vLLM 运行时：`src/omniav_detect/evaluation/vllm_runtime.py`
- 指标计算：`src/omniav_detect/evaluation/metrics.py`
- 输出与可视化：`src/omniav_detect/evaluation/outputs.py`、`visualization.py`
- 进度条：`src/omniav_detect/evaluation/progress.py`

## 关键调用关系

- `scripts/prepare_swift_av_sft.py` -> `omniav_detect.data.prepare_runner.main`
- `scripts/eval_batch_binary_qwen_omni.py` -> `omniav_detect.evaluation.batch_runner.main`
- `scripts/eval_batch_binary_qwen_omni_vllm.py` -> `omniav_detect.evaluation.batch_runner.main`

批量评估内部再分两条后端：

- `batch_runner` -> `python -m omniav_detect.evaluation.parallel_cli`
- `batch_runner` -> `python -m omniav_detect.evaluation.vllm_cli`

并行评估后端内部再调用：

- `parallel_runner` -> `python -m omniav_detect.evaluation.worker_cli`

因此，`scripts/` 目录现在只保留两个对外评估入口；其余 CLI 都下沉到 `src/omniav_detect/evaluation/`。

## 数据流

### 数据准备

1. `prepare_runner` 读取 `configs/data/swift_av_sft.yaml`
2. 根据 `--dataset` 只处理一个数据集
3. 扫描本地真实视频文件，过滤缺失、空文件、非法扩展名
4. 按配置选择默认分层切分或 MRDF subject-independent 5 折切分
5. 构造 ms-swift JSONL record
6. 如需独立音频输入，可在 JSONL 生成后批量抽音频并补写 `audios`
7. 写出训练/验证 JSONL、统计文件、缺失文件报告和预览样本

显式 `audios` 训练约束：

- 如果 JSONL 已经显式写入 `audios`，训练时必须关闭 `use_audio_in_video`。
- 否则同一条样本的音频会同时来自 `audios` 字段和视频内自动抽取，导致输入重复。

### MVAD 独立预处理

1. `mvad/run_prepare_mvad.sh` 调用 `python -m mvad.prepare_mvad`
2. `unzip_archives` 递归解压公开 `train/**/*.zip`，可按需跳过坏包并写入 manifest
3. `build_index_and_split` 扫描解压后视频，按目录推断 `real_real`、`real_fake`、`fake_real`、`fake_fake`
4. 按 `group_id` 执行 internal train/val 划分，避免同源组跨 split
5. `pairing` 优先按同目录同 stem 的视频和音频文件配对
6. 同目录找不到音频时，用 `ffprobe` 判断视频是否有内嵌音轨；有音轨则在同目录抽出同名 `.wav`
7. 同目录音频和内嵌音轨都不存在时，样本写入 `missing_audio_pairs.jsonl`，不进入训练 JSONL
8. `build_av_jsonl` 使用 index 中的原始音频路径或抽取目标路径写出显式 `audios`
9. 输出 `mvad_binary_train_with_audio.jsonl` 和 `mvad_binary_val_with_audio.jsonl`

### 并行评估

1. `batch_runner` 读取 `configs/eval/qwen_omni_binary_batch_eval.yaml`
2. 为每个 run 生成独立子进程命令
3. `parallel_runner` 将输入 JSONL 按 round-robin 切成多个 shard
4. 每个 GPU 对应一个 worker 子进程
5. 每个 worker 运行 `binary_logits.py` 的单 worker 评估逻辑
6. 汇总 `predictions.jsonl` / `bad_samples.jsonl`
7. 重算整体 Acc、AUC、AP/mAP、Confusion Matrix、Fake recall、Real recall

### vLLM 评估

1. `batch_runner` 读取 `configs/eval/qwen_omni_binary_batch_eval_vllm.yaml`
2. 为每个 run 生成 vLLM 子进程命令
3. `binary_logits_vllm.py` 读取 JSONL 并构造统一 prompt
4. 通过 `vllm_runtime.py` 处理 audio-video 输入和 logprob 提取
5. 写出预测、坏样本、指标和可视化结果

## 配置流

- 数据准备默认配置：`configs/data/swift_av_sft.yaml`
- 并行评估默认配置：`configs/eval/qwen_omni_binary_batch_eval.yaml`
- vLLM 评估默认配置：`configs/eval/qwen_omni_binary_batch_eval_vllm.yaml`

命令行可覆盖的核心参数包括：

- `output_root`
- `batch_size`
- `max_samples`
- `fps`
- `save_every`

## 输出文件

数据准备输出：

- `data/swift_sft/fakeavceleb/*.jsonl`
- `data/swift_sft/mavosdd/*.jsonl`
- `dataset_scan_summary.json`
- `dataset_stats.json`
- `missing_or_invalid_files.csv`
- `preview_samples.json`
- `missing_audio_pairs.jsonl`

评估输出：

- `predictions.jsonl`
- `bad_samples.jsonl`
- `metrics.json`
- `visualizations/`
- `batch_eval_summary.json`
- `batch_eval_summary.csv`

并行评估额外输出：

- `parallel_manifest.json`
- `parallel_status.json`
- `workers/worker_*/`

## 待确认

- conda 环境名
- Python 版本
- 服务器上的精确依赖版本
- vLLM 在当前服务器上的最终推荐并行参数
