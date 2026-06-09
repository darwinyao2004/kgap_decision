# 逐页中文讲稿

## Slide 1: Title
大家好，我今天讲的是这个知识缺口动作决策项目。这个项目的核心是把该不该回答这个问题单独拿出来建模。也就是说，系统看到一个问题、上下文、检索证据和候选回答之后，要先判断下一步应该直接回答、追问用户、检索更多证据、拒绝确认，还是先纠正问题里的错误前提。

## Slide 2: Talk Plan
今天的汇报我会从五个方面展开，我会先说这个问题为什么重要，然后讲数据和标签是怎么设计的。中间会花比较多时间讲代码实现，后面再看实验结果、消融和错误分析，最后讲一下现在这个实验能说明什么、不能说明什么。

## Slide 3: Problem: Answering Is Only One Possible Action
我们其实已经习惯在和大模型交互的时候每个问题都会得到大模型的直接回答，但是真实问答系统里，直接回答只是其中一种动作。比如用户没说预算和用途，那更合理的是追问更多细节；比如让模型判断医疗、法律、金融这类高风险问题时，证据不够就不能给确定建议；还有比如问题本身带了错误前提，系统还应该先把前提纠正掉。所以这个项目的重点是把这些动作显式建模出来。

## Slide 4: Task Definition
我们的任务输入由四个模块组成：分别是用户问题、对话上下文、检索证据和候选答案。输出有两个标签，一个是 `gap_type`，解释现在的信息缺口是什么；另一个是 `gold_action`，表示系统下一步具体做什么。这个action可以包括直接回答问题、...

## Slide 5: Evaluation Target
我们评价的主要指标是 action macro-F1，因为动作类别不止一个，而且不能只照顾多数类，所以我们会把所有类别的F1取一个平均来作为macro-F1。除此之外，我还算了 wrong-answer rate 和综合效用，我们定义的这个效用结合了正确性、错误惩罚项、过度拒绝、过度询问等等这些情况，这里对错误回答惩罚比较重，因为在这个任务里，最危险的不是“不够会说”，而是在不该回答的时候编出一个确定答案。

## Slide 6: Related Work Used as Background
相关工作主要给这个项目提供任务了一些边界。AmbigQA 和 ASQA 对应含混问题；FreshLLMs 对应动态事实和搜索；SelfCheckGPT 对应一致性和幻觉检测；Self-RAG 对应把检索作为可控动作；Sufficient Context 和 abstention benchmarks 则对应证据是否足够、什么时候应该停下来不答。这个项目不是复现其中某一篇，而是把这些边界合成一个动作决策任务。

## Slide 7: Dataset Overview
当前 full run 是 800 条样本。训练集 483，验证集 156，测试集 161。我们这里没直接用原始分组而是按 `group_id` 的模来切分的，因为同一个 group 里可能有同源改写，所以要避免它们跨训练集和测试集出现。

## Slide 8: Label Distribution
这页是标签分布。左边是 gap_type，右边是 action。可以看到 answer 不是主导类别，retrieve 和 abstain 的样本很多。这是有意设计的，因为这个实验就是要让模型面对“不能直接回答”的情况。如果数据里大部分都能直接答，那动作决策就没有太大意义。

## Slide 9: Record Schema
schema 这块在 `schema.py` 里。它定义了 `GapType`、`Action`、所有必需字段和校验函数。每条样本有文本字段，也有监督字段。需要注意的是，像 `risk_level`、`evidence_sufficiency_label` 这些字段虽然存在，但不能随便直接当特征喂给模型，否则会变成标签泄漏。

## Slide 10: Controlled Gap Types
标签类型覆盖七类情况。信息足够时 answer；问题含混或缺用户条件时 ask；证据缺失或时间敏感时多数需要 retrieve；如果用户问的是“证据能不能支持某个结论”，证据不够时也可能是 abstain；错误前提对应 challenge；高风险专业判断对应 abstain。这里最容易混的是 retrieve 和 abstain 的边界。

## Slide 11: Repository Structure
这页是我的工程结构。`knowledge_gap_decision/` 是核心代码；`data/processed/` 放生成后的数据、切分和特征；`results/` 放指标、预测、消融和图；`reports/` 是自动生成的报告；`tests/` 里有 schema、划分泄漏、特征和指标相关测试。

## Slide 12: Data Construction Code
数据构造主要在 `data_build.py`。`build_deepseek_dataset` 负责调用 DeepSeek 的 JSON 模式并修复生成结果；`build_dataset` 是离线 fallback；`validate_dataset` 做 schema 校验；`split_by_group` 做按 group 的切分；`write_dataset` 最后把 JSONL 和 label distribution 写到磁盘。这里的关键是让生成、校验、去重、切分都自动化。

## Slide 13: Feature Extraction Code
特征抽取集中在 `features.py` 的 `compute_features`。问题侧特征包括长度、时间词、条件词、主观词、实体数量和风险关键词。证据侧特征包括 TF-IDF 相似度、token overlap、coverage ratio 和 contradiction proxy。模型侧特征现在是离线伪采样的一致性特征。`feature_columns` 负责在消融时开关 evidence 或 uncertainty features。

## Slide 14: Model Code
模型主要在 `models.py`。`DualClassifier` 会分别训练 gap_type 分类器和 action 分类器，支持 logistic regression、random forest 和 GBDT。`TextClassifier` 是只看文本的 TF-IDF baseline。`prompted_llm_baseline` 让 DeepSeek 直接输出 JSON 决策。`full_method_predict` 是 Full Method：先用随机森林预测，再加少量保守的安全覆盖。

## Slide 15: Experiment Orchestration
主流程在 `run_experiment.py` 的 `run` 函数里。它会生成数据、加载 split、探测 API、选择 Full Method 配置、抽取特征、跑所有 baseline、写 metrics、画图、做消融、做显著性检验，再写错误分析和重复 seed 稳定性结果。也就是说，最终报告里的数字基本都是从这个入口自动生成的。

## Slide 16: Evaluation Code
评价逻辑在 `evaluate.py`。`evaluate_prediction` 算 macro-F1、accuracy、retrieval F1、contradiction accuracy 和 utility。`utility_components` 里面把 wrong answer、over-refusal、over-asking、turn cost 拆开算。`significance_rows` 做配对 bootstrap 和 McNemar 检验。这样结果不只是一个排行榜，也能看差异是否稳定。

## Slide 17: Question Ranker
追问生成在 `question_ranker.py`。`generate_candidates` 根据 required slots 生成候选问题，`score_question` 按 slot coverage、不确定性降低、具体性、可回答性、礼貌性来打分，同时惩罚诱导性问题。这个模块更像交互质量增强，不是 action-F1 提升的主要来源。

## Slide 18: Implementation Caveat
这一页我需要主动说明。当前 self-consistency 特征不是实际调用 LLM 多次采样，而是离线伪样本，而且伪样本生成函数会根据 `gap_type` 走不同分支。所以它更像一个受控代理特征，不能说成真实部署可用的不确定性估计。后面消融里它的贡献很大，但解释时要谨慎。

## Slide 19: Main Result: Action Macro-F1
主结果看 action macro-F1。Always Answer 和 Always Ask 基本不行，RAG threshold 和 self-consistency baseline 有一点提升，Prompted LLM Baseline 到 0.65 左右。传统结构化模型明显更强，Random Forest、GBDT 和 Full Method 都在 0.94 附近。Full Method 最高，是 0.945。

## Slide 20: Utility and Wrong-answer Risk
这张图把 wrong-answer rate 和 utility 放在一起看。Always Answer 的错误回答率最高，所以效用很低。Prompted LLM Baseline 的 macro-F1 还可以，但仍然有 0.118 的 wrong-answer rate。Full Method 的 wrong-answer rate 是 0，效用最高。这个结果说明，安全指标能揭示单纯 F1 看不到的一些问题。

## Slide 21: Headline Numbers
核心数字可以简单记这几个。Full Method 的 action macro-F1 是 0.945，action accuracy 是 0.938，utility 是 0.875，wrong-answer rate 是 0。Always Answer 的 action macro-F1 只有 0.052，utility 是 -2.404。Prompted LLM Baseline 的 wrong-answer rate 是 0.118，说明直接 prompt LLM 做判断并不够稳。

## Slide 22: Per-action F1
逐动作 F1 可以看出每类动作的表现。Full Method 整体比较均衡。Prompted LLM Baseline 在某些类别上还可以，但 abstain 不够稳。Logistic Regression 已经有很强表现，Random Forest 和 GBDT 更接近 Full Method。这个结果也说明当前任务里结构化特征非常有效。

## Slide 23: Action Confusion Matrix
动作混淆矩阵里，大部分预测落在对角线上。残余错误主要在 retrieve 和 abstain 的边界，比如“应该继续找证据”还是“基于当前证据拒绝确认”；还有少量 false premise 和 time-sensitive 的边界。这些都是之后构造 hard examples 时应该重点补的地方。

## Slide 24: Gap-type Confusion Matrix
gap_type 比 action 更细，所以更难。比如 ambiguous_question 和 user_info_missing 最终都可能是 ask，但解释类型不同；evidence_missing 和 time_sensitive 都可能是 retrieve，但原因不同。所以 gap_type 的混淆不一定都会导致系统动作错误，但它会影响解释和后续策略。

## Slide 25: Validation Selection
Full Method 不是只固定一个配置，它会在验证集上选一个变体。当前选中的是 no_evidence_side_features。这个结果有点有意思，说明当前验证集上证据侧启发式不一定最稳，问题侧和模型侧信号可能已经足够强。不过这个也提示我们，证据特征还有改进空间。

## Slide 26: Ablation
消融里最明显的是去掉 self-consistency 或 model uncertainty features，utility 下降大约 0.572，action macro-F1 下降大约 0.328。去掉 gap classifier、evidence sufficiency verifier 或 question ranker 的影响比较小。这里我要再次强调，uncertainty 特征的贡献大，但它目前是离线代理，所以不能过度宣传。

## Slide 27: Significance Tests
显著性检验用配对 bootstrap 和 McNemar。Full Method 相比 Always Answer、Always Ask、RAG threshold、Prompted LLM、Logistic Regression 和 Text Classifier 都有明显优势。但相比 Random Forest 和 GBDT，置信区间跨 0，差异不显著。所以更准确的说法是：Full Method 达到了最好的数值，但主要能力来自结构化模型本身。

## Slide 28: Repeated Split Stability
这里是三个随机 seed 的重复划分。action macro-F1 大概在 0.806 到 0.836，utility 在 0.802 到 0.816。波动不算大，说明在同一生成协议下结果还比较稳定。但这不代表真实用户分布下也稳定，因为真实问题会更乱，标签边界也会更模糊。

## Slide 29: Residual Errors
错误分析主要是其他动作混淆，还有少量 time-sensitive 没触发 retrieve。具体看案例，很多错误都集中在相邻动作边界，比如 retrieve vs abstain、false premise vs time-sensitive。这其实是好事，说明下一步应该补的不是更多简单样本，而是更难的边界样本。

## Slide 30: Example Boundaries
这里我用几个口头例子串一下。技术问题如果缺环境，就先问；政策、价格、截止日期这种动态事实要先查；如果问题本身假设错了，要先纠正前提；如果用户问“这段证据能不能证明某个结论”，证据不支持时应该拒绝确认。这个项目就是把这些判断标准化成可评估的动作标签。

## Slide 31: What the Results Say
结果可以总结成四点。第一，默认回答非常危险。第二，结构化特征在这个受控数据集上比直接 prompt LLM 更强。第三，Full Method 和 Random Forest、GBDT 很接近，所以不要把它说成远超传统模型。第四，不确定性风格的特征很有用，但当前实现还需要替换成真实的不确定性估计。

## Slide 32: Limitations
局限性主要有四个。数据是按 scenario 生成的，不是真实用户日志；标签可能带有生成器风格；证据冲突现在只是启发式 proxy，不是真正强 NLI；还有 self-consistency 不是实际 LLM 多次采样。这些限制不影响项目作为课程实验的完整性，但会影响结论能否外推。

## Slide 33: Next Steps
下一步我会优先做三件事。第一，引入真实用户问题，并人工复核标签。第二，把离线伪一致性换成真实多次采样、模型置信度或校准后的不确定性。第三，专门构造 hard cases，尤其是 retrieve vs abstain、false premise vs time-sensitive 这种边界。最后还要做跨领域、跨模型的 OOD 测试。

## Slide 34: Takeaway
最后的 takeaway 很简单：可靠 QA 系统不应该一上来就回答，而应该先判断当前信息是否支持回答。这个项目做了一个可复现的原型，把这个判断变成了数据、模型和评估流程。它的价值在于把问题定义清楚，也把目前的限制暴露清楚。

## Slide 35: References
最后是参考文献。这里列的论文主要是帮助定义任务边界，不是说本项目完整复现了它们。AmbigQA/ASQA 对应含混问题，FreshLLMs 对应动态事实，SelfCheckGPT 对应一致性思路，Self-RAG 对应检索控制，Sufficient Context 和 AbstentionBench 对应证据充分性和拒答边界。
