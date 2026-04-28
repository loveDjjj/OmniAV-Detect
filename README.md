# OmniAV-Detect

## 为 Qwen2.5-Omni SFT 准备 FakeAVCeleb 和 MAVOS-DD

使用 `scripts/prepare_swift_av_sft.py` 可以把本地 FakeAVCeleb 和 MAVOS-DD 视频文件转换成 ms-swift/Qwen2.5-Omni SFT 可直接使用的 JSONL 数据。脚本会扫描磁盘上的真实文件，写入视频绝对路径，跳过缺失、大小为 0 或扩展名不受支持的视频，并把跳过的样本记录到 `missing_or_invalid_files.csv`。

最小 dry-run 检查命令：

```bash
python scripts/prepare_swift_av_sft.py --dry_run --num_preview 5
```

同时生成 binary SFT 和 FakeAVCeleb structured SFT 的正式运行命令：

```bash
python scripts/prepare_swift_av_sft.py \
  --fakeavceleb_root /data/OneDay/FakeAVCeleb \
  --mavos_root /data/OneDay/MAVOS-DD \
  --output_dir /data/OneDay/OmniAV-Detect/data/swift_sft \
  --mode both
```

`data/swift_sft/` 下的主要输出文件：

- `fakeavceleb_binary_train.jsonl`, `fakeavceleb_binary_eval.jsonl`
- 当使用 `--mode structured` 或 `--mode both` 时生成 `fakeavceleb_structured_train.jsonl`, `fakeavceleb_structured_eval.jsonl`
- `mavosdd_binary_train.jsonl`, `mavosdd_binary_validation.jsonl`
- `mavosdd_binary_test_indomain.jsonl`, `mavosdd_binary_test_open_model.jsonl`
- `mavosdd_binary_test_open_language.jsonl`, `mavosdd_binary_test_open_full.jsonl`
- `mavosdd_binary_test_all.jsonl`
- `dataset_scan_summary.json`, `dataset_stats.json`, `missing_or_invalid_files.csv`, `preview_samples.json`

每一行 JSONL 都遵循 ms-swift 的对话数据格式，包含 `messages`、`videos` 和 `meta`。Binary SFT 样本使用如下提问：

```text
<video>
Given the video, please assess if it's Real or Fake? Only answer Real or Fake.
```

assistant 的回答严格为 `Real` 或 `Fake`。FakeAVCeleb structured 样本会在 assistant 消息中放入一个紧凑 JSON 字符串，字段包括 `overall_label`、`video_label`、`audio_label`、`modality_type` 和 `evidence`。

Binary SFT 示例命令：

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
  --model Qwen/Qwen2.5-Omni-7B \
  --train_type lora \
  --dataset /data/OneDay/OmniAV-Detect/data/swift_sft/fakeavceleb_binary_train.jsonl \
  --val_dataset /data/OneDay/OmniAV-Detect/data/swift_sft/fakeavceleb_binary_eval.jsonl \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --save_steps 500 \
  --eval_steps 500 \
  --output_dir /data/OneDay/OmniAV-Detect/outputs/qwen2_5_omni_fakeavceleb_binary
```

如果要训练 MAVOS-DD，把 `--dataset` 和 `--val_dataset` 分别替换为 `mavosdd_binary_train.jsonl` 和 `mavosdd_binary_validation.jsonl`。SFT 结束后，可以使用 MAVOS-DD 的 test JSONL 文件做独立测试集评估。
