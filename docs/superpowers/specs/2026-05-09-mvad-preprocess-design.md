# MVAD Preprocess Design

**目标**

在仓库根目录新增独立的 `mvad/` 数据预处理管线，用于把服务器上的 MVAD 公开 `train` 数据整理成 Qwen2.5-Omni 可直接训练的显式音频 JSONL 数据集。流程覆盖 zip 解压、视频索引、按同源组防泄漏划分 train/val、音频抽取、JSONL 生成，以及对应的一键 bash 命令。

**范围**

- 支持处理用户当前已下载的 MVAD 公开 `train` 目录。
- 输出二分类 `Real/Fake` 的 Qwen2.5-Omni SFT JSONL。
- 训练输入采用显式 `audios`，不走视频内自动抽音频。
- 划分方式为 internal `train/val`，不声称等同于论文官方 `test`。

**非目标**

- 本轮不接入 `prepare_swift_av_sft.py` 的统一数据准备入口。
- 本轮不实现 MVAD 的 structured 标注格式。
- 本轮不实现 stage2 训练和官方 test 复现。

## 1. 数据假设与标签规则

### 1.1 当前目录假设

MVAD 公开数据当前只有 `train`，目录包含四类模态组合：

- `train/real_real`
- `train/real_fake`
- `train/fake_real`
- `train/fake_fake`

其中：

- `real_real` 视为二分类 `Real`
- `real_fake`、`fake_real`、`fake_fake` 统一视为二分类 `Fake`

同时保留四类原始模态标签，用于统计和切分检查。

### 1.2 输入物理形态

当前原始下载内容以 `.zip` 为主，因此预处理必须先解压，再扫描真实视频文件。支持的视频扩展名沿用当前仓库公共约束：

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.webm`

### 1.3 baseline 定位

本次输出只能定义为：

- `MVAD public train-only subset internal validation baseline`

不能定义为：

- `MVAD paper official test benchmark`

原因是当前公开数据缺少论文中的官方 `test` split。

## 2. 目录与文件设计

新增独立目录 `mvad/`，结构如下：

- `mvad/README.md`
  - 说明 MVAD 数据准备流程、目录约定、命令示例。
- `mvad/common.py`
  - 路径工具、标签映射、视频扫描、分组键归一化。
- `mvad/unzip_archives.py`
  - 递归发现 zip 并解压，生成解压清单。
- `mvad/build_index_and_split.py`
  - 扫描解压后视频，构建样本索引，执行 group-aware train/val 划分，输出索引与统计。
- `mvad/build_av_jsonl.py`
  - 从索引抽取音频，生成带 `audios` 的 Qwen JSONL。
- `mvad/prepare_mvad.py`
  - 串联上述步骤的一体化 CLI。
- `mvad/run_prepare_mvad.sh`
  - 一键执行预处理全流程。
- `mvad/train_stage1_MVAD.sh`
  - 使用生成后的 train JSONL 训练 Qwen2.5-Omni stage1 baseline。

新增测试文件：

- `tests/test_mvad_prepare.py`

新增输出目录约定：

- `/data/OneDay/OmniAV-Detect/data/mvad_unpacked`
- `/data/OneDay/OmniAV-Detect/data/mvad_processed`
- `/data/OneDay/OmniAV-Detect/data/audio_cache/mvad`
- `/data/OneDay/OmniAV-Detect/data/swift_sft/mvad`

## 3. 数据流设计

### 3.1 Step A: 解压 zip

输入：

- MVAD 原始下载根目录，例如 `/data/MVAD`

处理：

- 递归查找 `train/**/*.zip`
- 每个 zip 解压到 `mvad_unpacked/` 下的稳定相对路径目录
- 默认跳过已经成功解压且带完成标记的目录
- 产出解压 manifest，记录 zip 路径、输出目录、状态、文件数

输出：

- `unpack_manifest.json`

### 3.2 Step B: 建立视频索引

处理：

- 递归扫描解压目录中的真实视频文件
- 从相对路径推断：
  - `modality_type`
  - `overall_label`
  - `video_source`
  - `audio_source`
  - `generation_path`，例如 `direct` 或 `indirect`

单条样本 meta 至少包含：

- `dataset="MVAD"`
- `overall_label`
- `video_label`
- `audio_label`
- `modality_type`
- `relative_path`
- `video_source`
- `audio_source`
- `group_id`

### 3.3 Step C: group-aware train/val 划分

核心约束：

- 同一“内容组”只能落在 train 或 val 的一边

推荐默认：

- `val_ratio=0.1`
- `seed=42`

#### group 规则

当前公开目录没有统一 metadata，因此按启发式生成 `group_id`：

- `real_real`
  - 以 `stem` 为主，前缀加上数据源目录，避免不同数据源同名冲突
- `fake_real`
  - 以 `stem` 为主，每条样本单独成组
- `real_fake`
  - 以“原始真实视频内容”成组
  - 从文件名中去除音频生成器后缀，如 `AudioX`、`FoleyCrafter`、`HunYuan`、`MMAudio`
- `fake_fake/indirect`
  - 以“同一个假视频主体 + 不同音频生成器版本”成组
  - 同样去除音频生成器后缀
- `fake_fake/direct`
  - 直接生成的视频音频一体样本通常没有派生音频版本，按 `stem` 单独成组

这是启发式规则，不宣称等同于官方 identity split，因此会把以下产物写出供人工抽查：

- `group_manifest.jsonl`
- `split_stats.json`
- `split_preview.json`

### 3.4 Step D: 抽取音频并生成 JSONL

对 split 后的视频索引：

- 用 `ffmpeg` 抽取音频
- 默认输出：
  - `wav`
  - `16k`
  - `mono`

生成两份 JSONL：

- `mvad_binary_train_with_audio.jsonl`
- `mvad_binary_val_with_audio.jsonl`

每条记录格式与当前仓库保持一致：

- `messages`
- `videos`
- `audios`
- `meta`

prompt 使用显式音频版本：

- user prompt 包含 `<video>` 和 `<audio>`
- assistant 只输出 `Real` 或 `Fake`

## 4. CLI 与 bash 设计

### 4.1 `prepare_mvad.py`

提供一体化入口，支持：

- `--source_root`
- `--unpack_root`
- `--work_root`
- `--audio_root`
- `--jsonl_root`
- `--val_ratio`
- `--seed`
- `--ffmpeg`
- `--overwrite`
- `--skip_unzip`
- `--skip_audio`
- `--dry_run`

### 4.2 `run_prepare_mvad.sh`

串联：

1. 解压
2. 索引
3. 划分
4. 抽音频
5. 生成 JSONL

### 4.3 `train_stage1_MVAD.sh`

训练策略参考当前 FakeAVCeleb/MAVOS-DD 的 stage1：

- `swift sft`
- `tuner_type=lora`
- `use_audio_in_video=False`
- 数据集指向 `mvad_binary_train_with_audio.jsonl`

同时建议把验证 JSONL 预留给后续评估脚本。

## 5. 错误处理

需要显式处理以下失败情况：

- zip 解压失败
- zip 解压后没有视频
- 视频扩展名不合法
- 重复文件名导致目标路径冲突
- ffmpeg 抽音频失败
- 划分后某个模态类型在 val 中为 0

策略：

- 不静默吞错
- 写 manifest 和统计
- 对关键错误返回非零退出码

## 6. 测试设计

新增 `tests/test_mvad_prepare.py`，覆盖：

- zip 解压清单生成
- 路径到四类模态标签映射
- `group_id` 归一化规则
- group-aware 划分不泄漏
- JSONL 生成时带 `audios`
- dry-run 时不实际调用 ffmpeg，但路径规划正确

测试样本使用临时目录和小假文件，不依赖真实 MVAD 数据。

## 7. 交付物

本轮交付完成后，你将得到：

- 独立的 `mvad/` 预处理目录
- 一键预处理脚本
- 一键 stage1 训练脚本
- internal train/val JSONL
- 音频缓存目录
- 解压、划分、抽音频的统计文件

## 8. 风险与约束

- 最大风险不是代码，而是公开 MVAD 只有 `train`，所以结果不能拿去和论文官方 test 直接对比。
- 第二个风险是 `group_id` 只能基于目录和文件名启发式构造，仍需抽查 split 结果是否存在近重复泄漏。
- 第三个风险是解压后实际文件命名可能和当前目录假设不完全一致，因此实现中必须把路径解析逻辑集中在一个模块，便于后续修正。
