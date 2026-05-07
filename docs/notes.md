# Notes

## 需求

基于 `reports/eval_mrdf5fold` 中 FakeAVCeleb MRDF5Fold fold1 的最新评估结果，生成一份只说明当前数据集构建方式和评估效果的简短 Markdown 报告。

## 修改文件

- reports/fakeavceleb_mrdf5fold_eval_report.md
- docs/notes.md
- docs/logs/2026-05.md

## 修改内容

- 新增 FakeAVCeleb MRDF5Fold fold1 数据构建与效果简报。
- 报告说明了 subject-independent 5-fold、二分类标签合并规则、显式音频输入设置。
- 汇总 Without SFT、Stage1、Stage1 -> Stage2 三组结果，并补充 balanced accuracy 与 Real recall，指出当前高 accuracy 主要来自测试集 Fake 占比 97.59%。

## 验证

```powershell
python - <<'PY'
from pathlib import Path
p = Path('reports/fakeavceleb_mrdf5fold_eval_report.md')
text = p.read_text(encoding='utf-8-sig')
assert 'FakeAVCeleb MRDF5Fold' in text
assert '97.59%' in text
assert 'Real recall' in text or 'Real Recall' in text
print('report ok')
PY
```

结果：通过，报告可用 UTF-8 正常读取，关键指标已写入。

## Git

- branch: `docs/fakeavceleb-mrdf5fold-eval-report`
- commit: `docs: add fakeavceleb mrdf5fold eval report`
