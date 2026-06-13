# 逐页中文讲稿

## Slide 1: Title
大家好，我今天汇报的是知识缺口动作决策实验。核心问题不是让模型直接生成长答案，而是在回答前先判断下一步动作：直接回答、追问、检索、拒答，还是纠正错误前提。

## Slide 2: Talk Plan
我会先介绍问题和任务定义，再讲数据、代码实现、实验结果、消融和错误分析，最后说明当前结论和局限。

## Slide 3: Problem: Answering Is Only One Possible Action
真实问答系统里，直接回答只是一个动作。缺用户条件时要追问，事实可能过期时要检索，高风险或证据不足时要拒绝确认，问题前提错误时要先纠正。

## Slide 4: Task Definition
输入包括用户问题、对话上下文、检索证据和候选答案。输出包括 `gap_type` 和 `gold_action`。系统真正执行的是 action，gap_type 主要用于解释信息缺口。

## Slide 5: Evaluation Target
主指标是 action macro-F1，同时看 wrong-answer rate 和综合效用。综合效用会重罚不该回答时直接回答，因为这是本任务里最危险的错误。

## Slide 6: Related Work Used as Background
AmbigQA/ASQA 对应含混问题，FreshQA 对应动态事实，SelfCheckGPT 对应一致性特征，Self-RAG 对应检索动作，Sufficient Context 和 AbstentionBench 对应证据充分性与拒答边界。

## Slide 7: Dataset Overview
full run 有 800 条样本，train/val/test 为 483/156/161。划分按 `group_id` 完成，避免同源改写样本同时进入训练集和测试集。

## Slide 8: Label Distribution
标签分布刻意包含较多 retrieve 和 abstain 场景，使任务不是“多数时候直接回答”。这能更好检验动作决策器是否会避免过早回答。

## Slide 9: Record Schema
`schema.py` 定义了标签、必需字段和校验逻辑。风险等级、证据充分性标签等字段用于标注和评估，不直接作为模型特征。

## Slide 10: Controlled Gap Types
七类 gap 覆盖信息足够、问题含混、缺用户条件、证据缺失、时间敏感、错误前提和高风险专业场景。最容易混淆的是 retrieve、abstain 和 challenge_premise 的边界。

## Slide 11: Repository Structure
核心代码在 `knowledge_gap_decision/`，数据在 `data/processed/`，结果在 `results/`，报告在 `reports/`，测试在 `tests/`。

## Slide 12: Data Construction Code
`build_deepseek_dataset` 调用 DeepSeek 生成受控样本，`build_dataset` 是 deterministic fallback，`validate_dataset` 做 schema 校验，`split_by_group` 避免 group 泄漏。

## Slide 13: Feature Extraction Code
特征包括问题侧特征、证据侧 TF-IDF/overlap/contradiction proxy，以及 DeepSeek 多次采样得到的 self-consistency 投票和理由一致性特征。

## Slide 14: Model Code
模型包括固定策略、RAG 阈值、自一致性启发式、Prompted LLM、Logistic Regression、Random Forest、GBDT、TF-IDF 文本分类器和 Full Method。

## Slide 15: Experiment Orchestration
`run_experiment.run` 负责从数据生成到指标、消融、显著性检验、错误分析和补充稳定性检查的全流程。稳定性检查现在明确标注为 fallback 模板数据检查。

## Slide 16: Evaluation Code
`evaluate_prediction` 计算 F1、accuracy、retrieval F1、contradiction accuracy 和 utility；`significance_rows` 做 paired bootstrap 和 McNemar 检验。

## Slide 17: Question Ranker
question ranker 只影响 ask 动作后的追问质量。它按槽位覆盖、不确定性降低、具体性、可回答性和礼貌性排序，不直接决定主分类分数。

## Slide 18: Self-consistency Feature Guardrails
self-consistency 每条样本采样五次，只给模型四个输入字段，不给任何 gold label。缓存里保存 input hash、action/rationale 和采样参数，不保存监督字段。

## Slide 19: Main Result: Action Macro-F1
当前测试集最强方法是 Logistic Regression，action macro-F1 为 0.747。Full Method 为 0.685，Prompted LLM Baseline 为 0.650。简单线性模型在当前特征和样本规模下最稳。

## Slide 20: Utility and Wrong-answer Risk
Always Answer 的 wrong-answer rate 是 0.851，utility 为 -2.404，说明默认回答很危险。Logistic Regression 的 utility 最高，为 0.493；Full Method 为 0.355。

## Slide 21: Headline Numbers
这页强调不要把 Full Method 写成最优。Full Method 是工程化组合基线；本次主结果应以 Logistic Regression 作为最佳监督基线。

## Slide 22: Per-action F1
Logistic Regression 在多个动作上更均衡。Full Method 的主要短板在 challenge_premise，经常把前提纠错处理成 abstain 或 retrieve。

## Slide 23: Action Confusion Matrix
Full Method 的主要错误包括 retrieve -> answer、challenge -> abstain、abstain -> ask。retrieve -> answer 是高成本错误，应优先压低。

## Slide 24: Gap-type Confusion Matrix
gap_type 是诊断标签，错了不一定改变系统动作。action 是最终执行标签，因此报告中应以 action 指标作为主线。

## Slide 25: Validation Selection
验证集选择了 `no_evidence_sufficiency_verifier`，说明当前 evidence sufficiency verifier 太粗，更像词面相关性判断，而不是可靠的证据蕴含判断。

## Slide 26: Ablation
去掉 self-consistency/uncertainty features 后 Full Method 掉分，说明 LLM 多采样特征有贡献。但直接 self-consistency baseline 很弱，投票结果更适合作为监督模型输入。

## Slide 27: Significance Tests
Full Method 明显强于固定策略和弱启发式，但相对 Logistic Regression 的差异为负且置信区间跨 0。统计结果支持克制结论。

## Slide 28: Fallback-template Split Check
这页不是 DeepSeek 主数据集的重复抽样，而是 fallback 模板数据的补充代码路径检查。它说明流程可复现，但不能证明主数据分布也同样稳定。

## Slide 29: Residual Errors
错误转移图比总分更有指导意义。下一步应补充 challenge vs abstain、retrieve vs answer、ask vs answer、高风险 ask vs abstain 的 hard cases。

## Slide 30: Example Boundaries
可以用几个例子解释边界：缺公司类型和银行时先 ask；今年政策或负责人先 retrieve；证据反驳前提时 challenge；高风险专业建议证据不足时 abstain。

## Slide 31: What the Results Say
主要 insight 是：动作决策可学；简单监督模型最强；Full Method 需要重新设计规则覆盖；Prompted LLM 有语义能力但容易过度回答；self-consistency 更适合作特征。

## Slide 32: Limitations
数据来自受控生成，不是真实用户日志；标签边界来自项目 policy；证据充分性判断仍是弱项；self-consistency 依赖 DeepSeek 和采样设置。

## Slide 33: Next Steps
下一步应把 Logistic Regression 作为主监督基线，做 utility-aware 校准；替换证据 verifier；增加 hard cases；设计 few-shot LLM baseline；引入真实用户问题和人工复核。

## Slide 34: Takeaway
最清楚的结论是：LLM 系统前面加一个简单、可审计的动作决策层是有价值的。当前最佳结果不是复杂 Full Method，而是线性监督模型。

## Slide 35: References
参考文献用于说明任务来源和边界。这个项目把相关工作中的含混、检索、拒答、证据充分性等问题压缩成一个更工程化的动作决策实验。
