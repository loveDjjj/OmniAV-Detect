#!/usr/bin/env bash
#
# 本脚本功能：
# - 面向 FakeAVCeleb 的 MRDF subject-independent 5 折 + 显式 audios 数据集；
# - 直接从 Qwen2.5-Omni 基模运行 Stage 2 only full tuning；
# - 冻结 LLM 主干，训练视觉编码器与 aligner。
#
# 使用方式：
# - FOLD_ID=1 bash train_stage2_FakeAVCeleb_MRDF5Fold_Audio.sh
# - 可用 MODEL_PATH / DATASET_PATH / OUTPUT_DIR 等环境变量覆盖默认路径。

set -euo pipefail

FOLD_ID="${FOLD_ID:-1}"
case "${FOLD_ID}" in
  1|2|3|4|5) ;;
  *)
    echo "FOLD_ID must be one of 1,2,3,4,5; got: ${FOLD_ID}" >&2
    exit 1
    ;;
esac

# 关键逻辑：当前 JSONL 已显式提供 `audios` 字段，必须关闭视频内重复抽音频。
export USE_AUDIO_IN_VIDEO=False
export use_audio_in_video=False
export ENABLE_AUDIO_OUTPUT=False
export TOKENIZERS_PARALLELISM=false
export MASTER_PORT="${MASTER_PORT:-29541}"
export PYTHONWARNINGS="ignore:PySoundFile failed:UserWarning,ignore:librosa.core.audio.__audioread_load:FutureWarning"

MODEL_PATH="${MODEL_PATH:-/data/OneDay/models/qwen/Qwen2.5-Omni-7B}"
DATASET_PATH="${DATASET_PATH:-/data/OneDay/OmniAV-Detect/data/swift_sft/fakeavceleb_mrdf5fold/fakeavceleb_mrdf5fold_fold${FOLD_ID}_binary_train_with_audio.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/OneDay/OmniAV-Detect/outputs/stage2_qwen2_5_omni_fakeavceleb_mrdf5fold_fold${FOLD_ID}_encoder_full_audio_explicit}"

echo "Using fold: ${FOLD_ID}"
echo "Using dataset: ${DATASET_PATH}"
echo "Writing stage2 output to: ${OUTPUT_DIR}"

NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 swift sft \
  --model "${MODEL_PATH}" \
  --model_type qwen2_5_omni \
  --tuner_type full \
  --freeze_llm true \
  --freeze_vit false \
  --freeze_aligner false \
  --vit_lr 1e-6 \
  --aligner_lr 2e-6 \
  --learning_rate 1e-6 \
  --dataset "${DATASET_PATH}" \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --max_length 1024 \
  --dataloader_num_workers 8 \
  --logging_steps 10 \
  --save_steps 100 \
  --save_total_limit 2 \
  --ddp_find_unused_parameters true \
  --freeze_parameters_regex ".*talker.*|.*token2wav.*" \
  --output_dir "${OUTPUT_DIR}"
