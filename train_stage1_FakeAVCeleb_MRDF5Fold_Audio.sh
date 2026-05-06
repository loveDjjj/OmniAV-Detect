#!/usr/bin/env bash
#
# 本脚本功能：
# - 面向 FakeAVCeleb 的 MRDF subject-independent 5 折 + 显式 audios 数据集；
# - 使用 Qwen2.5-Omni 做 stage1 LoRA 训练。
#
# 使用方式：
# - FOLD_ID=1 bash train_stage1_FakeAVCeleb_MRDF5Fold_Audio.sh

set -euo pipefail

FOLD_ID="${FOLD_ID:-1}"
case "${FOLD_ID}" in
  1|2|3|4|5) ;;
  *)
    echo "FOLD_ID must be one of 1,2,3,4,5; got: ${FOLD_ID}" >&2
    exit 1
    ;;
esac

# 关键逻辑：当前数据集已经显式提供 `audios` 字段，因此训练时必须关闭从视频重复抽音频。
export USE_AUDIO_IN_VIDEO=False
export use_audio_in_video=False
export ENABLE_AUDIO_OUTPUT=False
export TOKENIZERS_PARALLELISM=false
export MASTER_PORT="${MASTER_PORT:-29511}"
export PYTHONWARNINGS="ignore:PySoundFile failed:UserWarning,ignore:librosa.core.audio.__audioread_load:FutureWarning"

MODEL_PATH="${MODEL_PATH:-/data/OneDay/models/qwen/Qwen2.5-Omni-7B}"
DATASET_PATH="${DATASET_PATH:-/data/OneDay/OmniAV-Detect/data/swift_sft/fakeavceleb_mrdf5fold/fakeavceleb_mrdf5fold_fold${FOLD_ID}_binary_train_with_audio.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/OneDay/OmniAV-Detect/outputs/stage1_qwen2_5_omni_fakeavceleb_mrdf5fold_fold${FOLD_ID}_binary_audio_explicit}"

echo "Using fold: ${FOLD_ID}"
echo "Using dataset: ${DATASET_PATH}"
echo "Writing stage1 output to: ${OUTPUT_DIR}"

NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 swift sft \
  --model "${MODEL_PATH}" \
  --model_type qwen2_5_omni \
  --tuner_type lora \
  --dataset "${DATASET_PATH}" \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --max_length 1024 \
  --dataloader_num_workers 8 \
  --learning_rate 1e-5 \
  --logging_steps 10 \
  --save_steps 100 \
  --save_total_limit 2 \
  --output_dir "${OUTPUT_DIR}"
