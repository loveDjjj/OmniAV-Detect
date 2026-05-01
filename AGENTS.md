# AGENTS.md

## 环境

- conda 环境名：`待确认`
- Python 版本：`待确认`
- 主要依赖：见 `requirements.txt`

## 修改前先读

每次修改前必须先读：

1. `README.md`
2. `docs/notes.md`
3. `docs/architecture.md`
4. `docs/commands.md`
5. 当前需求涉及的代码文件
6. 当前需求涉及的配置文件

## 禁止乱改

- 不允许无需求重构核心算法。
- 不允许无需求改训练参数、数据路径、模型结构、损失函数、评估指标。
- 不允许随意移动目录或重命名文件。
- 不允许删除已有实验结果，除非用户明确要求。
- 不允许编造数据集路径、命令、指标或实验结果。

## 修改规范

- 代码尽量分模块，一个文件不应过长。
- 单个 Python 代码文件不应超过 500 行；超过时必须按职责拆分模块。
- 能复用已有函数就复用，不要复制粘贴重复逻辑。
- 新增或修改代码文件时，文件开头必须用中文注释说明该文件负责什么模块、主要函数/类有哪些、各自用途是什么。
- 主要函数必须写中文 docstring，说明函数功能、输入参数、输出结果和关键处理逻辑。
- 非主要函数可以简写注释，但必须说明用途。
- 关键代码块必须加中文注释，解释这段代码在做什么。
- 修改 YAML 配置时，新增项或重要修改项必须加注释。
- 不要为了“看起来高级”引入复杂抽象。

## 文档更新规则

每次实际修改后，必须同步更新：

1. `docs/notes.md`
2. `docs/logs/当前月份.md`
3. 如新增 / 修改运行命令，必须更新 `docs/commands.md`
4. 如新增 / 修改模块关系，必须更新 `docs/architecture.md`
5. 如新增依赖，必须更新 `requirements.txt`
6. 如新增输出文件类型，必须检查 `.gitignore`

## Git 规范

- 分支命名建议：`docs/...`、`fix/...`、`feat/...`、`exp/...`
- commit message 建议：
  - `docs: reorganize project documentation`
  - `fix: correct dataset loading path`
  - `feat: add evaluation script`
  - `exp: add baseline training config`

## 修改后输出格式

每次修改完成后，请输出：

1. 修改了哪些文件
2. 每个文件一句话说明
3. 是否运行验证命令
4. 验证结果
5. 建议的 branch 名称
6. 建议的 commit message
