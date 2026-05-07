#!/usr/bin/env bash
#
# 本脚本功能：
# - 评估 FakeAVCeleb MRDF subject-independent fold1 的 stage1->stage2 full checkpoint；
# - 输入 JSONL 使用显式 `audios` 字段，因此评估时关闭视频内重复抽音频。
#
# 使用方式：
# - bash eval_fakeavceleb_mrdf5fold_fold1_stage1_to_stage2_audio.sh

set -euo pipefail

export USE_AUDIO_IN_VIDEO=False
export use_audio_in_video=False
export ENABLE_AUDIO_OUTPUT=False
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore:PySoundFile failed:UserWarning,ignore:librosa.core.audio.__audioread_load:FutureWarning"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

MODEL_PATH="${MODEL_PATH:-/data/OneDay/OmniAV-Detect/outputs/stage1_to_stage2_fakeavceleb_mrdf5fold_fold1_encoder_full_audio_explicit}"
JSONL_PATH="${JSONL_PATH:-/data/OneDay/OmniAV-Detect/data/swift_sft/fakeavceleb_mrdf5fold/fakeavceleb_mrdf5fold_fold1_binary_test_with_audio.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/OneDay/OmniAV-Detect/outputs/batch_eval_binary/fakeavceleb_mrdf5fold_fold1_stage1_to_stage2_audio}"
GPUS="${GPUS:-0,1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
FPS="${FPS:-2.0}"

LATEST_MODEL_CKPT="$(find "${MODEL_PATH}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
if [[ -n "${LATEST_MODEL_CKPT}" ]]; then
  MODEL_PATH="${LATEST_MODEL_CKPT}"
fi

python -m omniav_detect.evaluation.parallel_cli \
  --model_path "${MODEL_PATH}" \
  --jsonl "${JSONL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --fps "${FPS}" \
  --gpus "${GPUS}" \
  --num_workers "${NUM_WORKERS}" \
  --no_use_audio_in_video
