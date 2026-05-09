#!/usr/bin/env bash
#
# 本脚本功能：
# - 一键执行 MVAD 公开 train 数据的 zip 解压、group-aware 划分、音频抽取和 Qwen JSONL 生成。
#
# 使用方式：
# - SOURCE_ROOT=/data/MVAD bash mvad/run_prepare_mvad.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOURCE_ROOT="${SOURCE_ROOT:-/data/MVAD}"
UNPACK_ROOT="${UNPACK_ROOT:-/data/OneDay/OmniAV-Detect/data/mvad_unpacked}"
WORK_ROOT="${WORK_ROOT:-/data/OneDay/OmniAV-Detect/data/mvad_processed}"
AUDIO_ROOT="${AUDIO_ROOT:-/data/OneDay/OmniAV-Detect/data/audio_cache/mvad}"
JSONL_ROOT="${JSONL_ROOT:-/data/OneDay/OmniAV-Detect/data/swift_sft/mvad}"
VAL_RATIO="${VAL_RATIO:-0.1}"
SEED="${SEED:-42}"
FFMPEG="${FFMPEG:-ffmpeg}"
FFPROBE="${FFPROBE:-ffprobe}"
EXTRACTOR="${EXTRACTOR:-7z}"

EXTRA_ARGS=()
if [[ "${OVERWRITE:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi
if [[ "${SKIP_UNZIP:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--skip_unzip)
fi
if [[ "${SKIP_AUDIO:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--skip_audio)
fi
if [[ "${SKIP_BAD_ARCHIVES:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--skip_bad_archives)
fi
if [[ "${ALLOW_EXTRACT_FROM_VIDEO:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--allow_extract_from_video)
fi
if [[ "${DRY_RUN:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--dry_run)
fi

cd "${REPO_ROOT}"
python -m mvad.prepare_mvad \
  --source_root "${SOURCE_ROOT}" \
  --unpack_root "${UNPACK_ROOT}" \
  --work_root "${WORK_ROOT}" \
  --audio_root "${AUDIO_ROOT}" \
  --jsonl_root "${JSONL_ROOT}" \
  --val_ratio "${VAL_RATIO}" \
  --seed "${SEED}" \
  --extractor "${EXTRACTOR}" \
  --ffmpeg "${FFMPEG}" \
  --ffprobe "${FFPROBE}" \
  "${EXTRA_ARGS[@]}"
