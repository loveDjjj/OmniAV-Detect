# Notes

## 需求

限制单个 Python 文件不超过 500 行，拆分 binary logits 评估模块，并增加评估可视化输出。

## 修改文件

- AGENTS.md
- README.md
- requirements.txt
- docs/notes.md
- docs/commands.md
- docs/architecture.md
- docs/logs/2026-05.md
- src/omniav_detect/evaluation/binary_logits.py
- src/omniav_detect/evaluation/constants.py
- src/omniav_detect/evaluation/data_io.py
- src/omniav_detect/evaluation/model_runtime.py
- src/omniav_detect/evaluation/metrics.py
- src/omniav_detect/evaluation/outputs.py
- src/omniav_detect/evaluation/visualization.py
- tests/test_eval_binary_logits_qwen_omni.py
- tests/test_project_structure.py

## 修改内容

- 在 `AGENTS.md` 增加单个 Python 文件不超过 500 行的规则，并用测试约束 `src/`。
- 将 `binary_logits.py` 拆分为常量、数据读取、模型运行、指标、输出和可视化模块。
- 评估输出新增 `visualizations/confusion_matrix.csv`、`score_distribution.csv`、`summary.html`，matplotlib 可用时额外写 PNG。

## 验证

```bash
python -B -m unittest tests.test_prepare_swift_av_sft tests.test_eval_binary_logits_qwen_omni tests.test_eval_batch_binary_qwen_omni tests.test_project_structure -v
python -B -m py_compile scripts/prepare_swift_av_sft.py scripts/eval_binary_logits_qwen_omni.py scripts/eval_batch_binary_qwen_omni.py src/omniav_detect/config.py src/omniav_detect/data/common.py src/omniav_detect/data/prepare_runner.py src/omniav_detect/data/fakeavceleb.py src/omniav_detect/data/mavosdd.py src/omniav_detect/evaluation/binary_logits.py src/omniav_detect/evaluation/constants.py src/omniav_detect/evaluation/data_io.py src/omniav_detect/evaluation/model_runtime.py src/omniav_detect/evaluation/metrics.py src/omniav_detect/evaluation/outputs.py src/omniav_detect/evaluation/visualization.py src/omniav_detect/evaluation/batch_runner.py
Get-ChildItem -Path src,scripts,tests -Recurse -Filter *.py | ForEach-Object { $count=(Get-Content -Encoding utf8 $_.FullName | Measure-Object -Line).Lines; [PSCustomObject]@{Lines=$count; Path=$_.FullName} } | Sort-Object Lines -Descending
```

结果：通过

## Git

- branch: `refactor/split-binary-eval-visualization`
- commit: `git commit -m "refactor: split binary evaluation and add visualizations"`
