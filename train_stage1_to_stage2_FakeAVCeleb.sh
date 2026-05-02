#!/usr/bin/env bash
#
# 本脚本功能：
# - 先将 FakeAVCeleb 的 stage1 LoRA checkpoint merge 到基模；
# - 再以 merge 后的完整权重为初始化，按 stage2 的冻结 LLM / 训练视觉与对齐模块思路继续训练。
#
# 使用方式：
# - bash train_stage1_to_stage2_FakeAVCeleb.sh
# - 如需覆盖默认路径，可在执行前导出同名环境变量，例如：
#   STAGE1_OUTPUT_DIR=/path/to/stage1_out bash train_stage1_to_stage2_FakeAVCeleb.sh

set -euo pipefail

export USE_AUDIO_IN_VIDEO=True
export use_audio_in_video=True
export ENABLE_AUDIO_OUTPUT=False
export TOKENIZERS_PARALLELISM=false
export MASTER_PORT="${MASTER_PORT:-29521}"
export PYTHONWARNINGS="ignore:PySoundFile failed:UserWarning,ignore:librosa.core.audio.__audioread_load:FutureWarning"

MODEL_PATH="${MODEL_PATH:-/data/OneDay/models/qwen/Qwen2.5-Omni-7B}"
DATASET_PATH="${DATASET_PATH:-/data/OneDay/OmniAV-Detect/data/swift_sft/fakeavceleb/fakeavceleb_binary_train.jsonl}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-/data/OneDay/OmniAV-Detect/outputs/stage1_qwen2_5_omni_fakeavceleb_binary_audio_in_video/checkpoint-472}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-/data/OneDay/OmniAV-Detect/outputs/stage1_to_stage2_fakeavceleb_merged}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-/data/OneDay/OmniAV-Detect/outputs/stage1_to_stage2_fakeavceleb_encoder_full}"

LATEST_STAGE1_CKPT="$(find "${STAGE1_OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
if [[ -z "${LATEST_STAGE1_CKPT}" ]]; then
  echo "No stage1 checkpoint found under: ${STAGE1_OUTPUT_DIR}" >&2
  exit 1
fi

echo "Using stage1 checkpoint: ${LATEST_STAGE1_CKPT}"
echo "Merging LoRA into: ${MERGED_MODEL_DIR}"

CUDA_VISIBLE_DEVICES=0 \
swift export \
  --adapters "${LATEST_STAGE1_CKPT}" \
  --merge_lora true \
  --output_dir "${MERGED_MODEL_DIR}"

echo "Starting stage2-style full tuning from merged model: ${MERGED_MODEL_DIR}"

NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 swift sft \
  --model "${MERGED_MODEL_DIR}" \
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
  --output_dir "${STAGE2_OUTPUT_DIR}"
