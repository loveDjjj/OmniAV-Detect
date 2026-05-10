#!/usr/bin/env bash
#
# 本脚本功能：
# - 使用 MVAD 显式 audios JSONL 训练 Qwen3-Omni-30B-A3B-Thinking stage1 LoRA baseline；
# - 训练 2 个 epoch，并显式关闭验证集划分和训练中评估。
#
# 使用方式：
# - bash mvad/train_stage1_MVAD_Qwen3Omni30BThinking.sh

set -euo pipefail

# 关键逻辑：MVAD JSONL 已包含 audios 字段，训练时必须关闭视频内重复抽音频。
export USE_AUDIO_IN_VIDEO=False
export use_audio_in_video=False
export ENABLE_AUDIO_OUTPUT=False
export TOKENIZERS_PARALLELISM=false
export MASTER_PORT="${MASTER_PORT:-29542}"
export PYTHONWARNINGS="ignore:PySoundFile failed:UserWarning,ignore:librosa.core.audio.__audioread_load:FutureWarning"

# Qwen3-Omni-30B 显存压力明显高于 Qwen2.5-Omni-7B，默认先用更低帧数和更小 batch 启动。
export FPS="${FPS:-0.5}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-25088}"
export MAX_PIXELS="${MAX_PIXELS:-501760}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_PATH="${MODEL_PATH:-/data/OneDay/models/Qwen3-Omni-30B-A3B-Thinking}"
DATASET_PATH="${DATASET_PATH:-/data/OneDay/OmniAV-Detect/data/swift_sft/mvad/mvad_binary_train_with_audio.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/OneDay/OmniAV-Detect/outputs/stage1_qwen3_omni_30b_a3b_thinking_mvad_binary_audio_explicit}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

echo "Using Qwen3-Omni model: ${MODEL_PATH}"
echo "Using MVAD dataset: ${DATASET_PATH}"
echo "Writing stage1 output to: ${OUTPUT_DIR}"
echo "Warning: Qwen3-Omni-30B-A3B-Thinking may OOM under ordinary DDP LoRA on 2x48GB GPUs."

NPROC_PER_NODE="${NPROC_PER_NODE}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" swift sft \
  --model "${MODEL_PATH}" \
  --model_type qwen3_omni \
  --tuner_type lora \
  --dataset "${DATASET_PATH}" \
  --split_dataset_ratio 0 \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --max_length "${MAX_LENGTH}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --learning_rate 1e-5 \
  --logging_steps 10 \
  --eval_strategy no \
  --save_steps 100 \
  --save_total_limit 2 \
  --gradient_checkpointing true \
  --output_dir "${OUTPUT_DIR}"
