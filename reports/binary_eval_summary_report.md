# Qwen2.5-Omni 音视频伪造检测阶段性结果分析

## 一、当前两阶段做法

我们当前复现的是两步训练：

| 阶段 | 做法 | 当前冻结/训练范围 | 重点说明 |
| --- | --- | --- | --- |
| Stage 1 | LoRA 指令微调 | 主要通过 LoRA 适配模型，让模型学会按 `Real/Fake` 输出 | 用于快速对齐二分类任务 |
| Stage 2 | 在 Stage 1 基础上继续 full tuning | <span style="color:#C00000"><strong>当前冻结 LLM 主干，只训练视觉编码器和 aligner</strong></span> | <span style="color:#C00000"><strong>论文只明确说打开音视频编码器，没有完全说清 LLM 主干是否也训练；这里可能是复现差异来源</strong></span> |

两个数据集的特殊设置：

| 数据集 | 当前设置 | 原因 |
| --- | --- | --- |
| FakeAVCeleb | 使用正常视频输入，没有额外 FPS/帧数限制 | 视频整体较短，当前主要问题不是算力，而是结果明显异常 |
| MAVOS-DD | `FPS=2.0`，`FPS_MAX_FRAMES=64`，并限制视频像素 | <span style="color:#C00000"><strong>MAVOS-DD 视频普遍更长，是当前 2x4090 下评估和训练耗时的主要瓶颈</strong></span> |

## 二、数据集划分与分布

### FakeAVCeleb

当前 FakeAVCeleb 使用 70% / 30% 切分，并把 `F-F`、`F-R`、`R-F`、`R-R` 都纳入二分类。只要音频或视频有伪造，就标为 `Fake`。

| 划分 | 总数 | Fake | Real | Fake 占比 | Real 占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 15080 | 14730 | 350 | 97.68% | 2.32% |
| Eval | 6464 | 6314 | 150 | 97.68% | 2.32% |

| 划分 | F-F | F-R | R-F | R-R |
| --- | ---: | ---: | ---: | ---: |
| Train | 7584 | 6796 | 350 | 350 |
| Eval | 3251 | 2913 | 150 | 150 |

<span style="color:#C00000"><strong>这个划分极度偏 Fake，预测全 Fake 的 accuracy baseline 就是 97.68%。因此当前 FakeAVCeleb 不能只看 accuracy，也需要看 Fake recall、Real recall 和 balanced accuracy。</strong></span>

### MAVOS-DD

MAVOS-DD 按官方思路包含 in-domain、open-language、open-model、open-full 等 split。当前训练集整体比 FakeAVCeleb 平衡，但不同测试 split 的真假比例差异很大。

| 划分 | 总数 | Fake | Real | 说明 |
| --- | ---: | ---: | ---: | --- |
| Train | 21370 | 11073 | 10297 | 基本平衡 |
| Validation | 3895 | 2180 | 1715 | 基本可用 |
| In-domain | 12408 | 7223 | 5185 | 已完成评估 |
| Open-language | 13330 | 5332 | 7998 | 已完成评估 |
| Open-model | 31551 | 21189 | 10362 | 评估尚未完整汇总 |
| Open-full | 4484 | 4484 | 0 | <span style="color:#C00000"><strong>统计异常，需要复核标签构造</strong></span> |

<span style="color:#C00000"><strong>MAVOS-DD 长视频是当前算力瓶颈。即使限制到 2 FPS 和最多 64 帧，open-model / open-full 的完整评估仍然很慢。</strong></span>

## 三、已有结果

### FakeAVCeleb

| 方法 | Accuracy | AUC | AP/mAP | Fake recall | Real recall | Balanced Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base / without SFT | 17.65% | 0.6290 | 0.9835 | 15.93% | 90.00% | 52.97% |
| Stage 1 LoRA | 30.21% | 0.6334 | 0.9850 | 29.00% | 81.33% | 55.17% |
| Stage 2 only | 18.84% | 0.6245 | 0.9833 | 17.18% | 88.67% | 52.93% |
| Stage 1 -> Stage 2 | 30.93% | 0.5666 | 0.9831 | 29.71% | 82.00% | 55.86% |

<span style="color:#C00000"><strong>FakeAVCeleb 的核心问题很明确：验证集 97.68% 是 Fake，但模型大量预测 Real，导致 accuracy 很低。Stage 1 和 Stage 1 -> Stage 2 有提升，但仍没有解决 Fake 漏检。</strong></span>

### MAVOS-DD

| Split | 状态 | Accuracy | AUC | AP/mAP | Fake recall | Real recall | Balanced Acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| In-domain | 完整 | 80.76% | 0.9503 | 0.9662 | 68.07% | 98.33% | 83.20% |
| Open-language | 完整 | 83.78% | 0.9388 | 0.9231 | 62.84% | 97.74% | 80.29% |
| Open-model | 未完整 | 65.59% | - | - | 47.92% | 97.24% | 72.58% |

<span style="color:#C00000"><strong>MAVOS-DD 结果比 FakeAVCeleb 正常，但同样偏向预测 Real：Real recall 很高，Fake recall 偏低。实际伪造检测中，这意味着漏检 Fake 的风险仍然明显。</strong></span>

## 四、与论文结果对比

为了直接看差距，下面把论文 AV-LMMDetect 结果和我们当前结果放在同一张表里。未完整跑完的结果单独标注，不作为最终结论。

| 数据集 / Split | 论文 Acc | 论文 AUC | 论文 mAP | 我们当前 Acc | 我们当前 AUC | 我们当前 AP/mAP | 当前状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FakeAVCeleb | 98.02% | 99.2 | - | 30.93% | 0.5666 | 0.9831 | Stage 1 -> Stage 2 已完成，但结果异常 |
| MAVOS-DD In-domain | 92.92% | 0.97 | 0.97 | 80.76% | 0.9503 | 0.9662 | 已完成 |
| MAVOS-DD Open-language | 85.58% | 0.90 | 0.88 | 83.78% | 0.9388 | 0.9231 | 已完成，最接近论文 |
| MAVOS-DD Open-model | 87.91% | 0.94 | 0.98 | 65.59% | - | - | <span style="color:#C00000"><strong>未完整跑完，仅 worker partial 结果</strong></span> |
| MAVOS-DD Open-full | 85.09% | 0.92 | 0.96 | - | - | - | <span style="color:#C00000"><strong>未跑完；且当前标签统计异常，需要先复核</strong></span> |

从表里可以看到：我们当前最接近论文的是 MAVOS-DD open-language，accuracy 83.78%，和论文 85.58% 差距不大；in-domain 的 AUC/mAP 接近论文，但 accuracy 明显低。<span style="color:#C00000"><strong>FakeAVCeleb 与论文差距最大，当前不是轻微复现误差，而是判别方向存在明显问题。</strong></span>

## 五、当前重点问题判断

我们的主要猜测如下：

1. <span style="color:#C00000"><strong>FakeAVCeleb 数据极度不平衡，且当前模型反而偏向预测 Real，这是最优先的问题。</strong></span> 这可能来自训练不足、prompt/token 校准、类别构造，或当前 Real/Fake 判决阈值不适合该分布。

2. <span style="color:#C00000"><strong>Stage 2 的冻结范围可能和论文不完全一致。</strong></span> 我们当前冻结 LLM，只训练视觉编码器和 aligner；如果论文实际也训练了语言侧或更多跨模态参数，就会造成明显差异。

3. <span style="color:#C00000"><strong>MAVOS-DD 的主要问题是长视频导致评估成本过高，open-model / open-full 还没有形成完整可靠结果。</strong></span> 当前 open-model 只能作为 partial 参考，不能作为最终指标。

4. <span style="color:#C00000"><strong>MAVOS-DD open-full 的标签统计异常必须先复核。</strong></span> 统计中出现 Fake=4484、Real=0，但生成方法里又包含 real，这说明数据构造或统计逻辑可能有问题。

5. <span style="color:#C00000"><strong>当前结果与论文差距不能只归因于 epoch 少。</strong></span> epoch 少确实会影响收敛，但 FakeAVCeleb 的大量 Fake 判 Real 更像是数据分布、判别阈值、训练策略或第二阶段解冻范围共同导致的问题。
