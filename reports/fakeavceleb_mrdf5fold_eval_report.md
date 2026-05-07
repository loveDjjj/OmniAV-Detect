# FakeAVCeleb MRDF5Fold 当前数据构建与评估简报

## 1. 当前数据集如何构建

当前 FakeAVCeleb 使用 MRDF 风格的 subject-independent 5-fold 划分，本次只分析 fold1。

构建流程如下：

- 扫描 FakeAVCeleb 四个模态目录：RealVideo-RealAudio、RealVideo-FakeAudio、FakeVideo-RealAudio、FakeVideo-FakeAudio。
- 二分类标签规则为：RealVideo-RealAudio 记为 Real，其余三类只要音频或视频任一模态为假，就统一记为 Fake。
- 使用每条样本的 subject id 匹配 MRDF 5-fold 的 train/test subject 列表，保证 train/test 按身份划分，不做随机视频级混合。
- 当前训练和评估都使用显式音频输入：JSONL 中同时包含视频路径和抽取后的音频路径，运行时关闭视频内部自动取音频，避免同一音频重复输入。

这个构建方式比普通 70/30 随机划分更接近 subject-independent 评估，但二分类标签合并后类别比例仍然极不均衡。

## 2. Fold1 测试集分布

| Split | Total | Fake | Real | Fake 占比 | Real 占比 |
|---|---:|---:|---:|---:|---:|
| fold1 test | 4147 | 4047 | 100 | 97.59% | 2.41% |

关键点：当前 fold1 测试集里 Fake 占 97.59%，所以一个始终预测 Fake 的模型也能得到 97.59% accuracy。因此这里不能只看 accuracy，必须同时看 Real recall 和 balanced accuracy。

## 3. 当前评估结果

| 模型设置 | Acc | AUC | AP/mAP | Fake Recall | Real Recall | Balanced Acc | 预测分布 |
|---|---:|---:|---:|---:|---:|---:|---|
| Without SFT | 6.49% | 0.6674 | 0.9870 | 4.20% | 99.00% | 51.60% | Fake 171 / Real 3976 |
| Stage1 | 97.59% | 0.7373 | 0.9903 | 100.00% | 0.00% | 50.00% | Fake 4147 / Real 0 |
| Stage1 -> Stage2 | 97.59% | 0.7213 | 0.9895 | 99.98% | 1.00% | 50.49% | Fake 4145 / Real 2 |

补充：三组评估都没有 bad samples，4147 条样本均带有显式音频路径。

## 4. 简单分析

当前结果的核心问题不是模型 accuracy 低，而是 accuracy 被测试集类别比例严重放大。

- Without SFT 基模更偏向预测 Real，所以 accuracy 很低，但 Real recall 达到 99%。
- Stage1 训练后模型完全倒向 Fake，accuracy 达到 97.59%，但 Real recall 为 0%。这个结果基本等价于多数类预测器。
- Stage1 -> Stage2 相比 Stage1 没有实质改善，只多预测出 2 个 Real，其中 1 个正确，Real recall 仍只有 1%。
- AUC 在 0.72 左右，说明分数排序可能有一定信息，但当前固定 Real/Fake token argmax 的判别阈值明显偏向 Fake。
- AP/mAP 接近 0.99 主要受 Fake 极大占比影响，不能单独作为“效果很好”的证据。

当前可以得出的结论：MRDF5Fold 解决了身份独立划分问题，但 fold1 的二分类测试集仍极度偏斜。现有 Stage1 和 Stage1 -> Stage2 的高 accuracy 不能说明模型真正学会了 FakeAVCeleb 检测，重点问题是 Real 类几乎无法识别。


