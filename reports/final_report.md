# 面向知识缺口的大语言模型动作决策：直接回答、主动追问与拒答的联合建模

## 1. 摘要
本实验把知识缺口场景下的问答决策拆成两个输出：缺口类型和回答动作。项目构造了 800 条统一格式样本，按 group_id 做 6:2:2 划分，比较固定动作、检索阈值、自一致性、结构化特征分类器、文本分类器和 Full Method。在当前 full 运行中，Full Method 的 action macro-F1 为 1.000，综合效用为 0.942；Always Answer 的 action macro-F1 为 0.092，效用为 -1.800。实验能说明规则化动作空间比直接回答更安全，但也暴露出一个现实问题：合成样本的模板痕迹较强，部分分类器拿到满分，数据难度还不够。

## 2. 研究背景与问题定义
真实问答系统经常面对信息缺失、问题含混、证据不足、知识过期、错误前提和高风险专业场景。直接生成答案并不总是合适：有时应主动追问用户条件，有时应检索最新证据，有时应拒绝给出专业判断，或先纠正问题中的错误前提。

本实验采用 T/P/E 框架定义任务：
- T：不确定条件下的回答动作选择。
- P：动作分类、错误回答率、过度拒答率、交互代价和综合效用。
- E：带缺口类型和动作标签的问答样本。

## 3. 相关工作简述
AmbigQA 和 ASQA 提醒我：很多问题并不是缺一个答案，而是问题本身有多种解释。FreshQA 处理动态事实，正好对应本实验里的 `time_sensitive -> retrieve`。SelfCheckGPT 用多次采样的一致性检查幻觉，我这里只做了离线扰动版，不能等同于真实 LLM 采样。Self-RAG 把检索纳入生成控制，本实验进一步把检索、追问、拒答和纠错都放进动作标签。Sufficient Context 和 AbstentionBench 相关工作则对应另两个边界：证据是否足够，以及模型什么时候应该停下来不答。

## 4. 数据构造
本次运行构造样本数为 800。原计划中的 AmbigQA、ASQA、FreshQA 没有直接下载进入主流程；为了保证离线可跑，数据主要由 fallback 模板和课程/技术/校园场景样本组成。每条样本包含 `user_initial_query`、`dialogue_context`、`retrieved_evidence`、`candidate_answer`、`gap_type`、`gold_action`、`required_slots`、证据标签、时间敏感标记、错误前提标记和风险等级。

划分使用固定随机种子 42，并按 `group_id` 划分，split 大小为 {'test': 160, 'train': 480, 'val': 160}。同一 group_id 只会进入一个 split，避免同源改写样本泄漏。

标签分布：
| label_type | label | count |
| --- | --- | --- |
| gap_type | sufficient_information | 218 |
| gap_type | ambiguous_question | 74 |
| gap_type | user_info_missing | 74 |
| gap_type | time_sensitive | 146 |
| gap_type | false_premise | 72 |
| gap_type | evidence_missing | 144 |
| gap_type | high_risk_or_expert_needed | 72 |
| gold_action | answer | 218 |
| gold_action | ask | 148 |
| gold_action | retrieve | 218 |
| gold_action | challenge_premise | 72 |
| gold_action | abstain | 144 |

## 5. 方法
整体流程包括：先标准化样本，再计算问题侧、证据侧和模型侧不确定性特征；随后训练 gap_type 分类器和 action 分类器，并在 Full Method 中使用规则优先级处理错误前提、高风险和时间敏感场景。申请书里写到的 BM25、embedding 相似度和 NLI 在当前实现中降级为 TF-IDF、token overlap、覆盖率和冲突启发式。Self-consistency 特征在 API 不可用时由离线模板扰动生成，记录在 `data/cache/llm_samples.jsonl`。

Full Method 的最终决策逻辑是：错误前提或证据冲突优先 `challenge_premise`，高风险优先 `abstain`，时间敏感优先 `retrieve`；如果用户问的是“资料能否证明/是否支持某结论”，而证据与候选肯定答案相冲突，则拒绝确认该结论；其他证据不足场景再根据用户条件缺失倾向选择 `ask` 或 `retrieve`。对 `ask` 样本，question utility ranker 生成候选追问，并按 slot coverage、预期不确定性降低、具体性、可回答性、礼貌性和诱导性惩罚排序。

## 6. 实验设置
运行模式：full。训练/验证/测试比例为 6:2:2，随机种子为 42。运行环境为 Python 3.14.3 / Darwin。GLM-5.1 API key 是否存在：True；API 是否通过严格 JSON 探测：False。本次 API 状态说明：API did not pass strict JSON probe; offline fallback used.

对照方法包括 Always Answer、Always Ask、RAG Similarity Threshold、Self-Consistency Baseline、Prompted LLM Baseline、Logistic Regression、Random Forest、GBDT、TF-IDF Text Encoder Classifier 和 Full Method。Prompted LLM Baseline 在 API 未通过时保留状态行但不参与有效比较。

## 7. 结果
总体指标如下：
| method | status | gap_type_macro_f1 | action_macro_f1 | action_accuracy | wrong_answer_rate | score |
| --- | --- | --- | --- | --- | --- | --- |
| Always Answer | ok | 0.066 | 0.092 | 0.300 | 0.700 | -1.800 |
| Always Ask | ok | 0.066 | 0.067 | 0.200 | 0 | -0.140 |
| RAG Similarity Threshold | ok | 0.381 | 0.397 | 0.438 | 0 | 0.321 |
| Self-Consistency Baseline | ok | 0.568 | 0.545 | 0.525 | 0 | 0.420 |
| Prompted LLM Baseline | skipped_api_unavailable | 0.066 | 0.092 | 0.300 | 0.700 | -1.800 |
| Logistic Regression | ok | 1 | 1 | 1 | 0 | 0.942 |
| Random Forest | ok | 1 | 1 | 1 | 0 | 0.942 |
| GBDT | ok | 1 | 1 | 1 | 0 | 0.942 |
| Text Encoder Classifier | ok | 1 | 1 | 1 | 0 | 0.942 |
| Full Method | ok | 1 | 1 | 1 | 0 | 0.942 |

Full Method 的 action macro-F1 为 1.000，action accuracy 为 1.000，wrong answer rate 为 0.000，综合效用为 0.942。相比 Always Answer，Full Method 的效用提升为 2.742；相比 RAG 阈值法，效用差值为 0.622。固定回答策略的错误主要来自把追问、检索、拒答和纠错样本都硬答掉。另一方面，Logistic Regression、Random Forest、GBDT 和文本分类器全部达到 1.0，这更像是数据模板清晰造成的“容易题”，不能被解读为模型已经解决真实开放问答。

主要类别指标保存在 `results/per_class_metrics.csv`。Full Method 的动作混淆矩阵见 `results/confusion_matrix_action.png`，缺口类型混淆矩阵见 `results/confusion_matrix_gap_type.png`，效用对比图见 `results/utility_comparison.png`。

## 8. 消融实验
| variant | action_macro_f1 | gap_type_macro_f1 | score | delta_action_macro_f1 | delta_score |
| --- | --- | --- | --- | --- | --- |
| Full - no gap type classifier | 1 | 1 | 0.942 | 0 | 0 |
| Full - no evidence sufficiency verifier | 1 | 1 | 0.942 | 0 | 0 |
| Full - no self-consistency features | 1 | 1 | 0.942 | 0 | 0 |
| Full - no question utility ranker | 1 | 1 | 0.942 | 0 | 0 |
| Full - no evidence-side features | 1 | 1 | 0.942 | 0 | 0 |
| Full - no model uncertainty features | 1 | 1 | 0.942 | 0 | 0 |

这一组消融没有改变动作分数。我的理解是：当前数据模板的提示词和标签边界过于清楚，随机森林动作分类器单独就能学到主要模式，所以额外规则、gap_type 输出和 self-consistency 特征没有在测试集上拉开差距。这不是“所有模块都没用”的证明，只说明现在的数据还不够难。

重复划分实验：
| seed | test_size | gap_type_macro_f1 | action_macro_f1 | action_accuracy | wrong_answer_rate | score |
| --- | --- | --- | --- | --- | --- | --- |
| 13 | 160 | 1 | 1 | 1 | 0 | 0.947 |
| 42 | 160 | 1 | 1 | 1 | 0 | 0.942 |
| 101 | 160 | 1 | 1 | 1 | 0 | 0.947 |

三组随机划分下，Full Method 的 action macro-F1 均值为 1.000，标准差为 0.000；综合效用均值为 0.945，标准差为 0.002。这个稳定性主要来自模板数据的规律性，不能外推为真实用户分布下同样稳定。

## 9. 显著性检验
| comparison | macro_f1_diff | ci_low | ci_high | p_bootstrap_two_sided | mcnemar_b01 | mcnemar_b10 | mcnemar_stat | mcnemar_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full Method vs Always Answer | 0.908 | 0.891 | 0.925 | 0 | 0 | 112 | 110.009 | 0 |
| Full Method vs Always Ask | 0.934 | 0.917 | 0.952 | 0 | 0 | 128 | 126.008 | 0 |
| Full Method vs RAG Similarity Threshold | 0.606 | 0.536 | 0.683 | 0 | 0 | 90 | 88.011 | 0 |
| Full Method vs Self-Consistency Baseline | 0.458 | 0.393 | 0.526 | 0 | 0 | 76 | 74.013 | 0 |
| Full Method vs Logistic Regression | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| Full Method vs Random Forest | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| Full Method vs GBDT | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| Full Method vs Text Encoder Classifier | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |

配对 bootstrap 给出 Full Method 与各 baseline 的 action macro-F1 差异及 95% 置信区间。McNemar 检验基于动作预测正确与否的配对列联表。若 quick 模式样本较少，置信区间可能较宽，部分差异不显著，应以 full 模式复核。

## 10. 错误分析
错误分析文件为 `results/error_analysis.md`。脚本现在只记录真实且去重后的错误，不再为了凑满 20 条复制案例。本次运行没有真实动作预测错误，因此只保留原因说明。

Full Method 在当前测试集没有动作预测错误。这个结果不代表真实场景已经解决，主要说明当前合成模板的边界比较清楚。后续应加入真实问句或更接近真实噪声的改写样本继续检查。

## 11. 结论
本实验完成了一个可复现的动作决策流程：数据构造、特征、模型、对照、消融、显著性检验和错误分析都能从脚本生成。当前结果支持一个比较朴素的结论：在知识缺口场景下，把“是否回答”拆成更细的动作标签，比默认直接回答更稳。更大的问题也很明显：模板数据太整齐，满分分类器说明任务区分度不足。后续最该补的不是再换一个模型，而是真实问句、噪声证据和更难的近邻标签样本。

## 12. 局限性
- 数据多为构造或半自动构造，真实用户分布可能不同；这也是分类器满分的主要原因。
- 高风险问题只做动作决策，不提供专业判断。
- API 不稳定或成本会影响 self-consistency 特征规模。
- 如果使用启发式 NLI proxy，其证据冲突判断能力有限。
- 当前 Prompted LLM Baseline 在 API 未通过严格 JSON 探测时不会强行运行，因此不能伪造与 GLM-5.1 的直接比较。

## 13. 可复现性说明
主要运行命令：
```bash
python scripts/check_zai_api_fixed.py
python -m pytest -q
python -m knowledge_gap_decision.run_experiment --quick
python -m knowledge_gap_decision.run_experiment --target-size 800
python -m knowledge_gap_decision.report
```
随机种子固定为 42。主要依赖见 `requirements.txt`。输出路径包括 `data/processed/`、`results/` 和 `reports/`。

## 14. 工具辅助说明
本项目使用了代码补全和文本整理工具辅助搭建脚本；报告中的数值、表格和图均来自本仓库实际运行结果。关键结论按 CSV/JSON 输出核对后写入，未手工改数。

## 15. 参考文献
- Min et al. 2020. AmbigQA: Answering Ambiguous Open-domain Questions. EMNLP.
- Stelmakh et al. 2022. ASQA: Factoid Questions Meet Long-Form Answers. EMNLP.
- Vu et al. 2023. FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation.
- Manakul et al. 2023. SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Black-Box LLMs.
- Asai et al. 2023. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
- Joren et al. 2024. Sufficient Context: A New Lens on Retrieval Augmented Generation Systems.
- AbstentionBench. 2025. Reasoning LLMs Fail on Unanswerable Questions.
