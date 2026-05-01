"""
本文件功能：
- 保存 Qwen2.5-Omni binary deepfake 评估中复用的提示词和标签常量。

主要内容：
- SYSTEM_PROMPT / USER_PROMPT：训练和评估保持一致的二分类问答提示词。
- LABELS：评估指标中固定使用的标签顺序。

使用方式：
- 被评估数据读取、模型运行、指标计算和输出模块共同引用。
"""

from __future__ import annotations


SYSTEM_PROMPT = "You are an audio-video deepfake detector. Given an input video, answer only Real or Fake."
USER_PROMPT = "<video>\nGiven the video, please assess if it's Real or Fake? Only answer Real or Fake."
USER_PROMPT_AFTER_VIDEO = "Given the video, please assess if it's Real or Fake? Only answer Real or Fake."
LABELS = ["Fake", "Real"]
