# Qwen2.5-Omni 二分类评估更新分析（eval_v1）

## 一、当前两阶段做法

当前复现流程仍然是两步：

| 阶段 | 做法 | 当前训练范围 | 风险点 |
| --- | --- | --- | --- |
| Stage 1 | LoRA 指令微调 | 用 LoRA 让模型学习只输出 `Real/Fake` | 主要用于任务对齐 |
| Stage 2 | Stage 1 后继续 full tuning | <span style="color:#C00000"><strong>冻结 LLM 主干，只训练视觉编码器和 aligner</strong></span> | <span style="color:#C00000"><strong>论文没有清楚说明 LLM 主干是否也参与 Stage 2 微调，这可能是复现差异来源</strong></span> |

两个数据集的设置差异：

| 数据集 | 当前设置 | 说明 |
| --- | --- | --- |
| FakeAVCeleb | 没有额外 FPS/帧数限制 | 当前主要问题不是速度，而是数据划分和判别方向异常 |
| MAVOS-DD | `FPS=2.0`，`FPS_MAX_FRAMES=64`，并限制视频像素 | <span style="color:#C00000"><strong>视频普遍较长，即使限制到 2 FPS 和最多 64 帧，完整测试仍然很慢</strong></span> |

## 二、数据集划分与分布

### FakeAVCeleb

当前 FakeAVCeleb 使用 70% / 30% 切分，并把 `F-F`、`F-R`、`R-F`、`R-R` 都纳入二分类。只要音频或视频有伪造，就标为 `Fake`。

| 划分 | 总数 | Fake | Real | Fake 占比 | Real 占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 15080 | 14730 | 350 | 97.68% | 2.32% |
| Eval | 6464 | 6314 | 150 | 97.68% | 2.32% |

<span style="color:#C00000"><strong>FakeAVCeleb 当前划分极度偏 Fake。预测全 Fake 的 accuracy baseline 就是 97.68%，所以当前结果不能只看 accuracy，必须同时看 Fake recall、Real recall 和 balanced accuracy。</strong></span>

### MAVOS-DD

MAVOS-DD 训练集整体比 FakeAVCeleb 平衡，但不同测试 split 的真假比例差异明显。

| 划分 | 总数 | Fake | Real | 说明 |
| --- | ---: | ---: | ---: | --- |
| Train | 21370 | 11073 | 10297 | 基本平衡 |
| Validation | 3895 | 2180 | 1715 | 基本可用 |
| In-domain | 12408 | 7223 | 5185 | 已完成测试 |
| Open-language | 13330 | 5332 | 7998 | 已完成测试 |
| Open-model | 31551 | 21189 | 10362 | 已完成测试 |
| Open-full | 4484 | 4484 | 0 | <span style="color:#C00000"><strong>标签统计异常：只有 Fake，没有 Real，AUC 无法正常解释</strong></span> |

## 三、eval_v1 已有完整结果

### FakeAVCeleb 消融结果

| 方法 | Accuracy | AUC | AP/mAP | Fake recall | Real recall | Balanced Acc | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Base / without SFT | 17.65% | 0.6290 | 0.9835 | 15.93% | 90.00% | 52.97% | 偏向预测 Real |
| Stage 1 LoRA | 30.21% | 0.6334 | 0.9850 | 29.00% | 81.33% | 55.17% | 有提升但仍很差 |
| Stage 2 only | 18.84% | 0.6245 | 0.9833 | 17.18% | 88.67% | 52.93% | 基本无改善 |
| Stage 1 -> Stage 2 | 30.93% | 0.5666 | 0.9831 | 29.71% | 82.00% | 55.86% | 当前最好，但仍异常 |

<span style="color:#C00000"><strong>FakeAVCeleb 的问题非常集中：验证集 97.68% 是 Fake，但模型大量预测 Real。完整 Stage 1 -> Stage 2 后仍只有 29.71% Fake recall，说明不是简单“没跑完”的问题。</strong></span>

### MAVOS-DD Stage 1 -> Stage 2 完整测试结果

| Split | 样本数 | Bad samples | Accuracy | AUC | AP/mAP | Fake recall | Real recall | Balanced Acc | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| In-domain | 12317 | 91 | 80.76% | 0.9503 | 0.9662 | 68.07% | 98.33% | 83.20% | 完整 |
| Open-language | 13327 | 3 | 83.78% | 0.9388 | 0.9231 | 62.84% | 97.74% | 80.29% | 完整 |
| Open-model | 31516 | 35 | 59.75% | 0.8433 | 0.9183 | 41.45% | 97.16% | 69.30% | 完整 |
| Open-full | 4483 | 1 | 54.29% | NaN | 1.0000 | 54.29% | 0.00% | 27.15% | <span style="color:#C00000"><strong>标签异常，只能作为数据问题提示</strong></span> |

MAVOS-DD 的整体结论比上一版更清楚：

- In-domain 和 open-language 与论文差距不算离谱，尤其 open-language 已经比较接近。
- Open-model 完整跑完后 accuracy 只有 59.75%，主要原因是 Fake recall 只有 41.45%，大量 Fake 被判成 Real。
- Open-full 不能按正常二分类结果分析，因为当前标签全是 Fake，AUC 为 NaN。
- <span style="color:#C00000"><strong>所有可用 MAVOS-DD split 都呈现同一个倾向：Real recall 很高，Fake recall 偏低，模型整体偏保守，容易漏检 Fake。</strong></span>

## 四、与论文 AV-LMMDetect 结果对比

| 数据集 / Split | 论文 Acc | 论文 AUC | 论文 mAP | 我们当前 Acc | 我们当前 AUC | 我们当前 AP/mAP | 当前状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FakeAVCeleb | 98.02% | 99.2 | - | 30.93% | 0.5666 | 0.9831 | 完整，但明显异常 |
| MAVOS-DD In-domain | 92.92% | 0.97 | 0.97 | 80.76% | 0.9503 | 0.9662 | 完整，有差距但 AUC/mAP 接近 |
| MAVOS-DD Open-language | 85.58% | 0.90 | 0.88 | 83.78% | 0.9388 | 0.9231 | 完整，最接近论文 |
| MAVOS-DD Open-model | 87.91% | 0.94 | 0.98 | 59.75% | 0.8433 | 0.9183 | 完整，但 Fake recall 很低 |
| MAVOS-DD Open-full | 85.09% | 0.92 | 0.96 | 54.29% | NaN | 1.0000 | <span style="color:#C00000"><strong>标签全 Fake，不能公平对比</strong></span> |

从完整结果看，当前结论可以更明确：

- MAVOS-DD open-language 最接近论文，说明当前流程并非完全无效。
- MAVOS-DD in-domain 的 AUC/mAP 接近论文，但 accuracy 低，说明固定 `Real/Fake` 判决下偏向 Real。
- MAVOS-DD open-model 差距明显，主要体现在 Fake recall 很低。
- FakeAVCeleb 与论文差距最大，更像是数据 protocol / 标签构造 / 判别策略问题，而不只是 epoch 数少。

## 五、当前重点判断

1. <span style="color:#C00000"><strong>FakeAVCeleb 优先怀疑数据划分或标签 protocol。</strong></span> 论文只说 70% / 30%，但没有说明是否把 `F-R`、`R-F` 都作为 Fake 参与同一个 binary protocol。我们当前这样处理后 Fake 占比达到 97.68%，非常不正常。

2. <span style="color:#C00000"><strong>Stage 2 冻结范围仍然是关键不确定项。</strong></span> 当前冻结 LLM 主干，只训练视觉编码器和 aligner；如果论文实际让 LLM 或更多跨模态参数参与训练，会直接影响复现结果。

3. <span style="color:#C00000"><strong>MAVOS-DD 的差距更可能来自训练不足和判决偏置。</strong></span> 我们目前 MAVOS-DD 只跑了 1 个 epoch，已完成 split 中 AUC/mAP 并不差，但 Fake recall 偏低，说明模型排序能力尚可，固定判决时偏 Real。

4. <span style="color:#C00000"><strong>MAVOS-DD open-full 必须先修数据再谈结果。</strong></span> 当前 open-full 全部标签都是 Fake，导致 AUC 为 NaN、Real recall 为 0，这不是正常测试集表现，而是数据构造或 split 解析需要复核。

5. <span style="color:#C00000"><strong>当前结果不能简单总结为“论文复现失败”。</strong></span> 更准确的说法是：MAVOS-DD 部分 split 已经接近论文，说明方法方向有一定有效性；FakeAVCeleb 和 MAVOS-DD open-full 暴露的是数据 protocol 与标签构造风险；open-model 暴露的是 Fake 漏检和泛化不足问题。
