#!/usr/bin/env bash
#
# 本脚本功能：
# - 使用 MVAD 显式 audios JSONL 训练 Qwen2.5-Omni stage1 LoRA baseline；
# - 训练 2 个 epoch，并显式关闭验证集划分和训练中评估。
#
# 使用方式：
# - bash mvad/train_stage1_MVAD.sh

set -euo pipefail

# 关键逻辑：MVAD JSONL 已包含 audios 字段，训练时必须关闭视频内重复抽音频。
export USE_AUDIO_IN_VIDEO=False
export use_audio_in_video=False
export ENABLE_AUDIO_OUTPUT=False
export TOKENIZERS_PARALLELISM=false
export MASTER_PORT="${MASTER_PORT:-29541}"
export PYTHONWARNINGS="ignore:PySoundFile failed:UserWarning,ignore:librosa.core.audio.__audioread_load:FutureWarning"

MODEL_PATH="${MODEL_PATH:-/data/OneDay/models/qwen/Qwen2.5-Omni-7B}"
DATASET_PATH="${DATASET_PATH:-/data/OneDay/OmniAV-Detect/data/swift_sft/mvad/mvad_binary_train_with_audio.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/OneDay/OmniAV-Detect/outputs/stage1_qwen2_5_omni_mvad_binary_audio_explicit}"

echo "Using MVAD dataset: ${DATASET_PATH}"
echo "Writing stage1 output to: ${OUTPUT_DIR}"

NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 swift sft \
  --model "${MODEL_PATH}" \
  --model_type qwen2_5_omni \
  --tuner_type lora \
  --dataset "${DATASET_PATH}" \
  --split_dataset_ratio 0 \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --max_length 1024 \
  --dataloader_num_workers 8 \
  --learning_rate 1e-5 \
  --logging_steps 10 \
  --eval_strategy no \
  --save_steps 100 \
  --save_total_limit 2 \
  --output_dir "${OUTPUT_DIR}"
