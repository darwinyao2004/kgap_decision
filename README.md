# 面向知识缺口的大语言模型动作决策

本项目把用户问题在不确定条件下的回答动作选择建模为机器学习任务，联合预测 `gap_type` 与 `gold_action`，比较直接回答、主动追问、检索、拒答和纠正错误前提等动作策略。

## 目录结构

- `knowledge_gap_decision/`：实验主代码。
- `data/raw/`：外部数据或缓存占位目录。
- `data/processed/`：统一数据集、训练/验证/测试划分和标签分布。
- `data/cache/`：LLM 或离线 self-consistency 采样缓存。
- `results/`：指标、预测、混淆矩阵、消融、显著性检验和错误分析。
- `reports/`：中文验收报告、Word 文档、展示提纲和实验日志。
- `scripts/`：命令行辅助脚本，保留 `scripts/check_zai_api_fixed.py`。
- `tests/`：schema、指标、划分泄漏和基础特征测试。

## 环境安装

```bash
python -m pip install -r requirements.txt
```

可选依赖如 `datasets`、`sentence-transformers`、`transformers`、`torch`、`rank_bm25`、`spacy` 不属于主流程必需项；缺失时会自动使用内置数据与轻量启发式 fallback。

## API 配置

GLM-5.1 调用从环境变量读取密钥：

```bash
export ZAI_API_KEY=...
python scripts/check_zai_api_fixed.py
```

项目代码复用该脚本的关键设置：`glm-5.1`、`httpx`、`http2=False`、`trust_env=False`、超时不少于 60 秒、结构化任务使用 `response_format={"type":"json_object"}`。代码和结果文件不会保存 API key 或 Authorization header。

如果 API 不可用，实验仍会运行离线 fallback，并在报告中记录。

## 快速运行

```bash
python -m pytest -q
python -m knowledge_gap_decision.run_experiment --quick
python -m knowledge_gap_decision.report
```

快速模式默认构造约 100 条样本，用于验收主流程。

## 完整实验

```bash
python -m knowledge_gap_decision.run_experiment --target-size 800
```

也可以使用：

```bash
make test
make quick
make full
make report
```

## 结果位置

- 数据：`data/processed/dataset.jsonl`、`train.jsonl`、`val.jsonl`、`test.jsonl`
- 指标：`results/metrics_summary.csv`、`results/per_class_metrics.csv`
- 消融：`results/ablation_summary.csv`
- 显著性检验：`results/significance_tests.csv`
- 预测与错误分析：`results/predictions_test.csv`、`results/error_analysis.md`
- 图表：`results/confusion_matrix_action.png`、`results/confusion_matrix_gap_type.png`、`results/utility_comparison.png`
- 报告：`reports/final_report.md`、`reports/final_report.docx`、`reports/presentation_outline.md`
