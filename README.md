# 面向知识缺口的大语言模型动作决策

本项目把用户问题在不确定条件下的回答动作选择建模为机器学习任务，联合预测 `gap_type` 与 `gold_action`，比较直接回答、主动追问、检索、拒答和纠正错误前提等动作策略。

## 目录结构

- `knowledge_gap_decision/`：实验主代码。
- `data/raw/`：外部数据或缓存占位目录。
- `data/processed/`：统一数据集、训练/验证/测试划分和标签分布。
- `data/cache/`：DeepSeek 数据生成、prompt baseline 和 self-consistency 采样缓存。
- `results/`：指标、预测、混淆矩阵、消融、显著性检验和错误分析。
- `reports/`：中文验收报告、Word 文档、展示提纲和实验日志。
- `scripts/`：命令行辅助脚本，包含 DeepSeek API 探测脚本。
- `tests/`：schema、指标、划分泄漏和基础特征测试。

## 环境安装

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

可选依赖如 `datasets`、`sentence-transformers`、`transformers`、`torch`、`rank_bm25`、`spacy` 不属于主流程必需项；缺失时会自动使用内置数据与轻量启发式 fallback。

## API 配置

DeepSeek 调用从环境变量读取密钥和模型名。当前项目会用 DeepSeek 做三件事：生成受控合成数据、运行 `Prompted LLM Baseline`，以及为 `sc_*` 特征做 5 次 action/rationale self-consistency 采样。

```bash
export DEEPSEEK_API_KEY=...
export DEEPSEEK_MODEL=deepseek-chat
.venv/bin/python scripts/check_deepseek_api.py
```

项目代码复用该脚本的关键设置：`httpx`、`http2=False`、`trust_env=False`、超时不少于 60 秒、结构化任务使用 `response_format={"type":"json_object"}`。默认接口为 `https://api.deepseek.com/chat/completions`，如需覆盖可设置 `DEEPSEEK_BASE_URL`。代码和结果文件不会保存 API key 或 Authorization header。

数据生成仍保留模板 fallback 以便构造样本，但完整实验的特征生成不允许离线替代 self-consistency。没有 `DEEPSEEK_API_KEY`、API 探测失败或批量调用失败时，实验会直接停止。Self-consistency 缓存写入 `data/cache/llm_self_consistency.jsonl`，每条缓存只包含四个输入字段对应的 input hash、action 投票和短理由，不包含 `gap_type`、`gold_action`、`final_answer`、`gold_clarifying_question` 或 `required_slots`。

## 快速运行

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m knowledge_gap_decision.run_experiment --quick
.venv/bin/python -m knowledge_gap_decision.report
```

快速模式默认构造约 100 条样本，用于验收主流程。
如果只想用离线模板构造数据，可加 `--offline-data`；但 self-consistency 特征和 LLM baseline 仍需要 DeepSeek API。

## 完整实验

```bash
.venv/bin/python -m knowledge_gap_decision.run_experiment --target-size 800
```

如果想强制重新调用 DeepSeek 生成数据，可加 `--refresh-llm-data`。

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
