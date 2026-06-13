# 面向知识缺口的大语言模型动作决策：直接回答、主动追问与拒答的联合建模

## 1. 摘要
本实验把知识缺口场景下的问答决策拆成两个输出：缺口类型和回答动作。项目构造了 800 条统一格式样本，按 group_id 做 6:2:2 划分，比较固定动作、检索阈值、自一致性、结构化特征分类器、文本分类器和 Full Method。在当前 full 运行中，最佳方法是 Logistic Regression，action macro-F1 为 0.747，综合效用为 0.493；Full Method 的 action macro-F1 为 0.685，综合效用为 0.355；Always Answer 的 action macro-F1 为 0.052，效用为 -2.404。实验能说明规则化动作空间比直接回答更安全；同时，Full Method 并不是本次主测试集上的最优方法，简单可审计的监督分类器更值得作为主基线。

## 2. 研究背景与问题定义
真实问答系统经常面对信息缺失、问题含混、证据不足、知识过期、错误前提和高风险专业场景。直接生成答案并不总是合适：有时应主动追问用户条件，有时应检索最新证据，有时应拒绝给出专业判断，或先纠正问题中的错误前提。

本实验采用 T/P/E 框架定义任务：
- T：不确定条件下的回答动作选择。
- P：动作分类、错误回答率、过度拒答率、交互代价和综合效用。
- E：带缺口类型和动作标签的问答样本。

## 3. 相关工作简述
AmbigQA 和 ASQA 提醒我：很多问题并不是缺一个答案，而是问题本身有多种解释。FreshQA 处理动态事实，正好对应本实验里的 `time_sensitive -> retrieve`。SelfCheckGPT 用多次采样的一致性检查幻觉，本实验也用 DeepSeek 多次采样得到动作投票和短理由一致性特征。Self-RAG 把检索纳入生成控制，本实验进一步把检索、追问、拒答和纠错都放进动作标签。Sufficient Context 和 AbstentionBench 相关工作则对应另两个边界：证据是否足够，以及模型什么时候应该停下来不答。

## 4. 数据构造
本次运行构造样本数为 800，生成模式为 `deepseek`。数据生成参考 AmbigQA/ASQA 的多解释问题、FreshQA 的动态事实、Sufficient Context 的证据充分性判断和 AbstentionBench 的拒答边界；DeepSeek 按受控 scenario 批量生成自然问题、对话上下文、检索证据和候选回答，代码负责类别配额、schema 校验、去重、group 划分和少量 fallback 补齐。每条样本包含 `user_initial_query`、`dialogue_context`、`retrieved_evidence`、`candidate_answer`、`gap_type`、`gold_action`、`required_slots`、证据标签、时间敏感标记、错误前提标记和风险等级。

划分使用固定随机种子 42，并按 `group_id` 划分，split 大小为 {'test': 161, 'train': 483, 'val': 156}。同一 group_id 只会进入一个 split，避免同源改写样本泄漏。

标签分布：
| label_type | label | count |
| --- | --- | --- |
| gap_type | sufficient_information | 137 |
| gap_type | ambiguous_question | 75 |
| gap_type | user_info_missing | 62 |
| gap_type | evidence_missing | 212 |
| gap_type | time_sensitive | 74 |
| gap_type | false_premise | 86 |
| gap_type | high_risk_or_expert_needed | 154 |
| gold_action | answer | 137 |
| gold_action | ask | 137 |
| gold_action | retrieve | 212 |
| gold_action | challenge_premise | 86 |
| gold_action | abstain | 228 |

## 5. 方法
整体流程包括：先标准化样本，再计算问题侧、证据侧和模型侧不确定性特征；随后训练 gap_type 分类器和 action 分类器，并在 Full Method 中使用验证集从少量预定义配置里选择最稳的组合。申请书里写到的 BM25、embedding 相似度和 NLI 在当前实现中降级为 TF-IDF、token overlap、覆盖率和冲突启发式。为避免标签泄漏，`evidence_sufficiency_label`、`false_premise_flag`、`time_sensitive_flag`、`risk_level` 不再直接作为模型特征；Self-consistency 特征由 DeepSeek 基于四个输入字段多次采样生成，记录在 `data/cache/llm_self_consistency.jsonl`，缓存条目不包含 `gap_type`、`gold_action`、`final_answer`、`gold_clarifying_question` 或 `required_slots`。

Full Method 的配置选择写入 `results/full_method_selection.json`，本次验证集选择为 `no_evidence_sufficiency_verifier`。最终决策以结构化特征分类器为主，只保留保守的安全覆盖；对 `ask` 样本，question utility ranker 生成候选追问，并按 slot coverage、预期不确定性降低、具体性、可回答性、礼貌性和诱导性惩罚排序。

## 6. 实验设置
运行模式：full。训练/验证/测试比例为 6:2:2，随机种子为 42。运行环境为 Python 3.14.3 / Darwin。deepseek 模型：deepseek-v4-flash；API key 是否存在：True；API 是否通过严格 JSON 探测：True。本次 API 状态说明：DeepSeek API available; LLM self-consistency features use cached or newly sampled model outputs. 如果没有 API key、探测失败或批量调用失败，实验会直接停止，不再使用离线回退特征。

对照方法包括 Always Answer、Always Ask、RAG Similarity Threshold、Self-Consistency Baseline、Prompted LLM Baseline、Logistic Regression、Random Forest、GBDT、TF-IDF Text Encoder Classifier 和 Full Method。Prompted LLM Baseline 的本次状态为 `ok`；LLM 调用失败会中止实验，而不是用默认动作补齐。

## 7. 结果
总体指标如下：
| method | status | gap_type_macro_f1 | action_macro_f1 | action_accuracy | wrong_answer_rate | score |
| --- | --- | --- | --- | --- | --- | --- |
| Always Answer | ok | 0.037 | 0.052 | 0.149 | 0.851 | -2.404 |
| Always Ask | ok | 0.057 | 0.056 | 0.161 | 0 | -0.190 |
| RAG Similarity Threshold | ok | 0.310 | 0.358 | 0.429 | 0 | 0.328 |
| Self-Consistency Baseline | ok | 0.306 | 0.287 | 0.422 | 0.006 | 0.297 |
| Prompted LLM Baseline | ok | 0.594 | 0.650 | 0.652 | 0.118 | 0.208 |
| Logistic Regression | ok | 0.668 | 0.747 | 0.758 | 0.062 | 0.493 |
| Random Forest | ok | 0.658 | 0.712 | 0.733 | 0.087 | 0.375 |
| GBDT | ok | 0.642 | 0.707 | 0.727 | 0.068 | 0.419 |
| Text Encoder Classifier | ok | 0.426 | 0.460 | 0.516 | 0.019 | 0.349 |
| Full Method | ok | 0.626 | 0.685 | 0.714 | 0.087 | 0.355 |

按综合效用排序，本次最佳方法是 Logistic Regression，action macro-F1 为 0.747，综合效用为 0.493。Full Method 的 action macro-F1 为 0.685，action accuracy 为 0.714，wrong answer rate 为 0.087，综合效用为 0.355。相比 Always Answer，Full Method 的效用提升为 2.759；相比 RAG 阈值法，效用差值为 0.028。固定回答策略的错误主要来自把追问、检索、拒答和纠错样本都硬答掉。新数据下 Prompted LLM Baseline、文本分类器和传统结构化模型之间拉开了差距，说明任务不再只是模板记忆；同时也说明 Full Method 的规则覆盖还需要重新设计，不能把它写成当前最强方法。

主要类别指标保存在 `results/per_class_metrics.csv`。Full Method 的动作混淆矩阵见 `results/confusion_matrix_action.png`，缺口类型混淆矩阵见 `results/confusion_matrix_gap_type.png`，效用对比图见 `results/utility_comparison.png`。

## 8. 消融实验
| variant | action_macro_f1 | gap_type_macro_f1 | score | delta_action_macro_f1 | delta_score |
| --- | --- | --- | --- | --- | --- |
| Full - no gap type classifier | 0.685 | 0.627 | 0.390 | -0.001 | 0.035 |
| Full - no evidence sufficiency verifier | 0.685 | 0.626 | 0.355 | 0 | 0 |
| Full - no self-consistency features | 0.610 | 0.590 | 0.294 | -0.075 | -0.061 |
| Full - no question utility ranker | 0.685 | 0.627 | 0.390 | -0.001 | 0.035 |
| Full - no evidence-side features | 0.665 | 0.577 | 0.305 | -0.021 | -0.050 |
| Full - no model uncertainty features | 0.610 | 0.590 | 0.294 | -0.075 | -0.061 |

`Full - no gap type classifier`、`Full - no question utility ranker` 的分数反而高于 Full Method，最高差值为 0.035，说明 Full Method 的人工规则在这批样本上有过度干预。`Full - no self-consistency features`、`Full - no model uncertainty features`、`Full - no evidence-side features` 会拉低分数，最大下降为 0.061，这部分模块对当前决策边界有实际贡献。`Full - no evidence sufficiency verifier` 没有造成可见变化。

重复划分补充检查：
| seed | test_size | gap_type_macro_f1 | action_macro_f1 | action_accuracy | wrong_answer_rate | score |
| --- | --- | --- | --- | --- | --- | --- |
| 13 | 160 | 0.882 | 0.862 | 0.912 | 0 | 0.857 |
| 42 | 160 | 0.914 | 0.840 | 0.925 | 0 | 0.866 |
| 101 | 160 | 0.910 | 0.874 | 0.912 | 0 | 0.857 |

这组结果使用内置 fallback 模板数据做补充检查，不是当前 DeepSeek 主数据集的重复抽样；因此只能说明代码流程在另一个可复现数据源上是否稳定，不能直接作为主结果置信度。三组划分下，Full Method 的 action macro-F1 均值为 0.859，标准差为 0.014；综合效用均值为 0.860，标准差为 0.004。该稳定性仍不能外推为真实用户分布下同样稳定。

## 9. 显著性检验
| comparison | macro_f1_diff | ci_low | ci_high | p_bootstrap_two_sided | mcnemar_b01 | mcnemar_b10 | mcnemar_stat | mcnemar_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full Method vs Always Answer | 0.630 | 0.551 | 0.701 | 0 | 5 | 96 | 80.198 | 0 |
| Full Method vs Always Ask | 0.626 | 0.543 | 0.698 | 0 | 8 | 97 | 73.752 | 0 |
| Full Method vs RAG Similarity Threshold | 0.325 | 0.230 | 0.423 | 0 | 14 | 60 | 27.365 | 0.000 |
| Full Method vs Self-Consistency Baseline | 0.397 | 0.305 | 0.486 | 0 | 9 | 56 | 32.554 | 0.000 |
| Full Method vs Prompted LLM Baseline | 0.036 | -0.048 | 0.116 | 0.368 | 17 | 27 | 1.841 | 0.175 |
| Full Method vs Logistic Regression | -0.062 | -0.141 | 0.010 | 0.100 | 21 | 14 | 1.029 | 0.310 |
| Full Method vs Random Forest | -0.026 | -0.060 | 0 | 0.102 | 3 | 0 | 1.333 | 0.248 |
| Full Method vs GBDT | -0.021 | -0.073 | 0.027 | 0.422 | 8 | 6 | 0.071 | 0.789 |
| Full Method vs Text Encoder Classifier | 0.227 | 0.116 | 0.332 | 0 | 21 | 53 | 12.986 | 0.000 |

配对 bootstrap 给出 Full Method 与各 baseline 的 action macro-F1 差异及 95% 置信区间。McNemar 检验基于动作预测正确与否的配对列联表。若 quick 模式样本较少，置信区间可能较宽，部分差异不显著，应以 full 模式复核。

## 10. 错误分析
错误分析文件为 `results/error_analysis.md`。脚本现在只记录真实且去重后的错误，不再为了凑满 20 条复制案例。下面列出自动抽样中的前几个案例：

错误类别计数：
- 其他动作混淆: 14
- ask 误判为 answer: 2
- abstain 误判为 answer: 1
- false premise compliance: 1
- retrieve 误判为 answer: 1
- answer 误判为 ask: 1

## 案例 1: 其他动作混淆
- query: Is it true that the UK has a law requiring all citizens to register their bicycles?
- gold: false_premise / challenge_premise
- predicted: evidence_missing / abstain
- key_features: {"ev_sufficiency_score": 0.0, "ev_contradiction_proxy": 1.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.621}
- 改进建议: 检查该类样本的标签边界，并补充相邻动作的对比样本。

## 案例 2: 其他动作混淆
- query: 清华大学今年是不是取消了文科专业？
- gold: false_premise / challenge_premise
- predicted: time_sensitive / retrieve
- key_features: {"ev_sufficiency_score": 0.041, "ev_contradiction_proxy": 1.0, "q_has_time_words": 1, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.443}
- 改进建议: 检查该类样本的标签边界，并补充相邻动作的对比样本。

## 案例 3: ask 误判为 answer
- query: 我需要开一个公司账户，需要准备哪些材料？
- gold: user_info_missing / ask
- predicted: sufficient_information / answer
- key_features: {"ev_sufficiency_score": 0.295, "ev_contradiction_proxy": 0.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.756}
- 改进建议: 先追问 公司类型、计划开户银行，再给出步骤或推荐。

## 11. 结论
本实验完成了一个可复现的动作决策流程：DeepSeek 数据生成、特征、模型、对照、消融、显著性检验和错误分析都能从脚本生成。当前结果支持一个比较朴素的结论：在知识缺口场景下，把“是否回答”拆成更细的动作标签，比默认直接回答更稳。本次主测试集上最强的是 Logistic Regression，Full Method 更适合作为工程化组合基线；不确定性/self-consistency 特征对 Full Method 有贡献，但人工规则覆盖仍需要重做。

## 12. 局限性
- 数据由 DeepSeek 按受控 scenario 生成，并非真实线上用户日志；真实用户分布可能不同。
- 数据生成仍可能出现轻微标签噪声，因此代码加入了 schema 校验、高风险一致性修复、去重和 fallback 补齐。
- 高风险问题只做动作决策，不提供专业判断。
- API 不稳定或成本会影响 self-consistency 特征规模；当前实现选择硬失败，避免静默回退污染结果。
- 如果使用启发式 NLI proxy，其证据冲突判断能力有限。
- 当前 Prompted LLM Baseline 和 self-consistency 采样都要求 DeepSeek API 可用；若批量预测出现解析失败，实验会停止。

## 13. 可复现性说明
主要运行命令：
```bash
.venv/bin/python scripts/check_deepseek_api.py
.venv/bin/python -m pytest -q
.venv/bin/python -m knowledge_gap_decision.run_experiment --quick
.venv/bin/python -m knowledge_gap_decision.run_experiment --target-size 800
.venv/bin/python -m knowledge_gap_decision.report
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
