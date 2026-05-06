# Notes

## 需求

基于 `reports/eval_v1` 中完整的 MAVOS-DD stage1->stage2 测试结果，结合旧版 `reports/binary_eval_summary_report.md`，新建一版更新分析报告。

## 修改文件

- reports/binary_eval_summary_report_v2.md
- docs/notes.md
- docs/logs/2026-05.md

## 修改内容

- 新建 `binary_eval_summary_report_v2.md`，保留一目了然的报告结构：两阶段做法、数据分布、eval_v1 完整结果、论文对比、重点判断。
- 更新 MAVOS-DD 四个 split 的完整 stage1->stage2 指标：in-domain、open-language、open-model、open-full。
- 明确 open-model 已完整跑完但 Fake recall 很低；open-full 标签全 Fake，AUC 为 NaN，不能公平对比论文。
- 保留 FakeAVCeleb 数据划分和结果异常判断。

## 验证

```bash
python -B -m py_compile src/omniav_detect/evaluation/metrics.py
python -c "from pathlib import Path; text=Path('reports/binary_eval_summary_report_v2.md').read_text(encoding='utf-8-sig'); assert 'MAVOS-DD Open-model' in text and '59.75%' in text and '标签全 Fake' in text"
```

结果：待运行

## Git

- branch: `docs/eval-v1-summary-report`
- commit: `git commit -m "docs: add eval v1 binary result analysis"`
