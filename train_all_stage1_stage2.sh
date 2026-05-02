#!/usr/bin/env bash
#
# 本脚本功能：
# - 顺序执行两个数据集的 stage1 -> stage2 训练脚本；
# - 将每个任务的标准输出与错误输出分别写入 logs/。

set -euo pipefail

mkdir -p logs

bash train_stage1_to_stage2_FakeAVCeleb.sh 2>&1 | tee logs/train_stage1_to_stage2_FakeAVCeleb.log
echo "train_stage1_to_stage2_FakeAVCeleb exit code: ${PIPESTATUS[0]}"

bash train_stage1_to_stage2_MAVOS-DD.sh 2>&1 | tee logs/train_stage1_to_stage2_MAVOS-DD.log
echo "train_stage1_to_stage2_MAVOS-DD exit code: ${PIPESTATUS[0]}"
