# 逐页中文讲稿

## Slide 1: Title
大家好，我今天讲的是这个知识缺口动作决策项目。这个项目的核心是把该不该回答这个问题单独拿出来建模。也就是说，系统看到一个问题、上下文、检索证据和候选回答之后，要先判断下一步应该直接回答、追问用户、检索更多证据、拒绝确认，还是先纠正问题里的错误前提。

## Slide 2: Talk Plan
今天的汇报我会从五个方面展开，我会先说这个问题为什么重要，然后讲数据和标签是怎么设计的。中间会用一点时间讲代码实现，后面再看实验结果、消融和错误分析，最后讲一下现在这个实验的结论和边界。

## Slide 3: Problem: Answering Is Only One Possible Action
我们其实已经习惯在和大模型交互的时候每个问题都会得到大模型的直接回答。真实问答系统里，直接回答只是其中一种动作。比如用户没说预算和用途，那更合理的是追问更多细节；比如让模型判断医疗、法律、金融这类高风险问题时，证据不够就需要暂缓给确定建议；还有比如问题本身带了错误前提，系统还应该先把前提纠正掉。所以这个项目的重点是把这些动作显式建模出来。

## Slide 4: Task Definition
我们的任务输入由四个模块组成：用户问题、对话上下文、检索证据和候选答案。输出有两个标签，一个是 `gap_type`，解释现在的信息缺口是什么；另一个是 `gold_action`，表示系统下一步具体做什么。动作集合包括 answer、ask、retrieve、abstain 和 challenge_premise。这里要强调一点：真正部署时系统执行的是 action，gap_type 更像诊断信息，用来解释为什么需要采取当前动作。

## Slide 5: Evaluation Target
我们评价的主要指标是 action macro-F1，因为动作类别不止一个，而且评价需要覆盖每个类别，所以我们会把所有类别的F1取一个平均来作为macro-F1。除此之外，我还算了 wrong-answer rate 和综合效用，这个效用结合了正确性、错误惩罚项、过度拒绝、过度询问等情况。这里对错误回答惩罚比较重，因为这个任务里最危险的情况，是模型在需要暂缓回答的时候编出一个确定答案。

## Slide 6: Related Work Used as Background
相关工作主要给这个项目提供了一些任务边界。AmbigQA 和 ASQA 对应含混问题；FreshLLMs 对应动态事实和搜索；SelfCheckGPT 对应一致性和幻觉检测；Self-RAG 对应把检索作为可控动作；Sufficient Context 和 abstention benchmarks 则对应证据是否足够、什么时候应该停下来不答。这个项目把这些边界合成了一个动作决策任务。

## Slide 7: Dataset Overview
当前 full run 是 800 条样本。训练集 483，验证集 156，测试集 161。这里按 `group_id` 的模来切分，因为同一个 group 里可能有同源改写，需要避免它们跨训练集和测试集出现。数据生成本身有缓存和模板兜底；self-consistency 特征来自真实 DeepSeek 多次采样。

## Slide 8: Label Distribution
这页是标签分布。左边是 gap_type，右边是 action。可以看到 answer 占比不高，retrieve 和 abstain 的样本很多。这个分布是有意设计的，因为这个实验要让模型面对大量“需要先采取其他动作”的情况。数据里存在足够多的非回答场景，动作决策才有实际意义。

## Slide 9: Record Schema
schema 这块在 `schema.py` 里。它定义了 `GapType`、`Action`、所有必需字段和校验函数。每条样本有文本字段，也有监督字段。需要注意的是，`risk_level`、`evidence_sufficiency_label` 这类字段服务于标注和评估，模型特征只使用部署时可见的信息。

## Slide 10: Controlled Gap Types
标签类型覆盖七类情况。信息足够时 answer；问题含混或缺用户条件时 ask；证据缺失或时间敏感时多数需要 retrieve；如果用户问的是“证据能不能支持某个结论”，证据不够时也可能是 abstain；错误前提对应 challenge；高风险专业判断对应 abstain。这里最容易混的是 retrieve 和 abstain 的边界。

## Slide 11: Repository Structure
这页是我的工程结构。`knowledge_gap_decision/` 是核心代码；`data/processed/` 放生成后的数据、切分和特征；`results/` 放指标、预测、消融和图；`reports/` 是自动生成的报告；`tests/` 里有 schema、划分隔离、特征和指标相关测试。

## Slide 12: Data Construction Code
数据构造主要在 `data_build.py`。`build_deepseek_dataset` 负责调用 DeepSeek 的 JSON 模式并规范化生成结果；`build_dataset` 是离线 fallback；`validate_dataset` 做 schema 校验；`split_by_group` 做按 group 的切分；`write_dataset` 最后把 JSONL 和 label distribution 写到磁盘。这里的关键是让生成、校验、去重、切分都自动化。

## Slide 13: Feature Extraction Code
特征抽取集中在 `features.py` 的 `compute_features`。问题侧特征包括长度、时间词、条件词、主观词、实体数量和风险关键词。证据侧特征包括 TF-IDF 相似度、token overlap、coverage ratio 和 contradiction proxy。模型侧特征现在来自 DeepSeek 多次采样：每次只看用户问题、对话上下文、检索证据和候选答案，然后输出 action 和一句短理由。`feature_columns` 负责在消融时开关 evidence 或 uncertainty features。

## Slide 14: Model Code
模型主要在 `models.py`。`DualClassifier` 会分别训练 gap_type 分类器和 action 分类器，支持 logistic regression、random forest 和 GBDT。`TextClassifier` 是只看文本的 TF-IDF baseline。`prompted_llm_baseline` 让 DeepSeek 直接输出 JSON 决策。`full_method_predict` 是 Full Method：先用随机森林预测，再加少量保守的安全覆盖。

## Slide 15: Experiment Orchestration
主流程在 `run_experiment.py` 的 `run` 函数里。它会生成数据、加载 split、探测 API、选择 Full Method 配置、抽取特征、跑所有 baseline、写 metrics、画图、做消融、做显著性检验，再写错误分析和重复 seed 稳定性结果。也就是说，最终报告里的数字基本都是从这个入口自动生成的。

## Slide 16: Evaluation Code
评价逻辑在 `evaluate.py`。`evaluate_prediction` 算 macro-F1、accuracy、retrieval F1、contradiction accuracy 和 utility。`utility_components` 里面把 wrong answer、over-refusal、over-asking、turn cost 拆开算。`significance_rows` 做配对 bootstrap 和 McNemar 检验。这样结果既有排行榜，也能看差异是否稳定。

## Slide 17: Question Ranker
追问生成在 `question_ranker.py`。`generate_candidates` 根据 required slots 生成候选问题，`score_question` 按 slot coverage、不确定性降低、具体性、可回答性、礼貌性来打分，同时惩罚诱导性问题。这个模块主要增强交互质量，action-F1 的提升主要来自前面的动作分类器。

## Slide 18: Self-consistency Feature Guardrails
这一页讲 self-consistency 特征的生成约束。每条样本调用 DeepSeek 五次，每次只提供用户问题、对话上下文、检索证据和候选答案四个输入字段。模型返回 action 和一句短理由，后续特征只统计投票熵、多数票比例、各动作票数和理由文本重合度。缓存文件只保存 input hash、四个输入字段名、五次 action/rationale 输出、采样温度和 schema 版本。API key、API 探测和 JSON 格式都是硬性前置条件。

## Slide 19: Main Result: Action Macro-F1
这页先看 action macro-F1。Always Answer 和 Always Ask 分数很低，说明固定策略解决不了这个任务；RAG threshold 和 self-consistency baseline 有一定提升，整体仍然停留在启发式水平。

Logistic Regression 在当前测试集最稳。它的 action macro-F1 是 0.747，高于 Random Forest 的 0.712、GBDT 的 0.707、Full Method 的 0.685，也高于 Prompted LLM Baseline 的 0.650。这说明在当前受控数据上，时间词、风险词、证据相关性和 self-consistency 投票这些特征已经提供了很强的线性信号。

这个结果给模型选择带来一个直接 insight：在样本量 800、特征维度不高、动作边界相对清楚的设置里，简单线性分类器的偏差和方差更合适。Full Method 的随机森林和安全覆盖规则提供了工程结构，当前测试集更支持简单线性模型。

## Slide 20: Utility and Wrong-answer Risk
这张图把 wrong-answer rate 和 utility 放在一起看，因为这个任务里错误类型比单纯分类对错更重要。Always Answer 的问题最明显，它把所有问题都硬答，所以 wrong-answer rate 是 0.851，utility 掉到 -2.404。这个基线说明，默认回答在知识缺口任务里是危险策略。

Logistic Regression 同时拿到最高 action macro-F1 和最高 utility：action macro-F1 是 0.747，utility 是 0.493，wrong-answer rate 是 0.062。Full Method 的 action macro-F1 是 0.685，utility 是 0.355，wrong-answer rate 是 0.087。Full Method 的错误分布带来更高的效用损失。

Prompted LLM Baseline 的 action macro-F1 是 0.650，utility 是 0.208，wrong-answer rate 是 0.118，高风险 unsafe answer rate 是 0.176。这个差异很有价值：DeepSeek 能识别不少语义边界，同时保留了聊天模型常见的 helpfulness bias，在证据不足或高风险场景里更容易给出 answer。

## Slide 21: Headline Numbers
这页把关键数字放在一起。最佳方法是 Logistic Regression，action macro-F1 是 0.747，utility 是 0.493。Full Method 的 action macro-F1 是 0.685，utility 是 0.355，wrong-answer rate 是 0.087。Prompted LLM Baseline 的 F1 是 0.650，wrong-answer rate 达到 0.118。

这里最重要的 insight 是模型复杂度和效果没有正相关。Full Method 通过随机森林加保守覆盖规则来提升安全性，当前测试集上表现低于更简单的线性模型。这个结果提醒我们，复杂 pipeline 需要用验证集和测试集同时证明收益。

另一个 insight 是 utility 和 macro-F1 的排序不同。Text Encoder Classifier 的 action macro-F1 只有 0.460，utility 有 0.349，接近 Full Method，因为它 wrong-answer rate 很低。Prompted LLM 的 macro-F1 高于 Text Encoder，utility 更低，因为它更容易提前回答。这个任务需要同时看 F1 和错误代价。

## Slide 22: Per-action F1
逐动作 F1 能看出每类方法的行为差异。Logistic Regression 比较均衡：retrieve 的 F1 是 0.841，abstain 是 0.747，challenge 是 0.735，answer 是 0.717，ask 是 0.694。它的优势来自各动作类别上的稳定表现。

Prompted LLM 的优势集中在 challenge_premise 和 retrieve 上，F1 分别是 0.783 和 0.747；abstain F1 是 0.492。它的 answer recall 是 0.875，abstain recall 是 0.340。这说明 DeepSeek 对明显矛盾和检索需求比较敏感，对“暂时不给结论”的 policy 边界执行较弱。

Full Method 的主要短板是 challenge_premise。它的 challenge F1 是 0.529，recall 是 0.429，经常把错误前提处理成 abstain 或 retrieve。它倾向于保守处理风险，前提纠错能力需要加强。

## Slide 23: Action Confusion Matrix
动作混淆矩阵展示的是 Full Method 的剩余错误。对角线是主要部分，关键错误集中在三个方向：retrieve 被预测成 answer 有 7 个，challenge 被预测成 abstain 有 8 个，abstain 被预测成 ask 有 5 个。

retrieve -> answer 是高成本错误，因为它意味着模型把证据不足或需要更新的信息当成可以直接回答。challenge -> abstain 的错误相对安全，会损失用户体验：系统知道当前不能直接答，却没有明确指出用户前提哪里错。abstain -> ask 说明高风险或不可验证问题有时被处理成继续收集信息。

这些错误比总分更能指导下一步。后续数据应该优先补这些相邻动作的对比样本，并且明确 policy hierarchy：什么时候触发 challenge，什么时候提高 retrieve 优先级，什么时候让高风险问题进入 abstain。

## Slide 24: Gap-type Confusion Matrix
gap_type 比 action 更细，所以它天然更难。比如 `ambiguous_question` 和 `user_info_missing` 最终都可能对应 ask，前者是问题有多种解释，后者是缺用户条件。再比如 `evidence_missing` 和 `time_sensitive` 最终都可能对应 retrieve，前者是证据不完整，后者是事实可能过期。

这个区别很重要，因为 gap_type 的错误有时不会改变动作。模型可能把 ambiguous_question 预测成 user_info_missing，只要 action 仍然是 ask，系统行为就是安全的。action 错误会直接影响系统输出，比如把 retrieve 预测成 answer。

所以这个项目里 action 是主任务，gap_type 更像诊断信息。汇报时可以强调：gap_type 帮我们分析动作背后的原因，最终系统真正执行的是 action。

## Slide 25: Validation Selection
这张图展示 Full Method 在验证集上的配置选择。图里同时放了 action macro-F1 和 utility，当前被选中的是 `no_evidence_sufficiency_verifier`，因为 selection 逻辑先看验证集 action macro-F1，再用 utility 作 tie-break。

这个结果说明当前 sufficiency verifier 太粗。它本质上还是 TF-IDF、overlap、coverage 和 contradiction proxy 的组合，主要判断相关性，对“这些证据是否足够支持候选答案完成用户任务”的刻画不足。验证集选择关闭这个 verifier，说明这条启发式规则的收益有限。

同时，完全去掉 evidence-side features 会降低验证集和测试集表现。测试消融里 no evidence-side features 的 action macro-F1 比 Full Method 低 0.021，utility 低 0.050。证据侧信号有价值，当前 verifier 需要换成更强的 entailment 或 answerability 判断。

## Slide 26: Ablation
消融结果显示 self-consistency 或 model uncertainty features 有稳定贡献。去掉这组特征后，action macro-F1 从 0.685 掉到 0.610，utility 从 0.355 掉到 0.294。

这说明真实 LLM self-consistency 是有用的特征组。当前实现每条样本调用 DeepSeek 五次，只用四个输入字段，让模型投 action 并给一句 rationale，然后计算 vote entropy、majority ratio、rationale overlap 和各动作票数。

同时，单独的 Self-Consistency Baseline action macro-F1 只有 0.287，说明 LLM 投票信号本身噪声很大。它更适合作为监督模型的输入特征，直接多数投票执行的效果较弱。

## Slide 27: Significance Tests
显著性检验进一步限制了我们能讲的结论。Full Method 相比 Always Answer、Always Ask、RAG threshold、Self-Consistency Baseline 和 Text Encoder Classifier 都有显著优势，这说明它确实强于很弱的固定策略和启发式策略。

Full Method 相比 Prompted LLM Baseline 的 action macro-F1 差只有 0.036，bootstrap 区间跨 0；相比 Logistic Regression、Random Forest 和 GBDT 的差异也落在统计不确定区间内。显著性检验支持一个克制结论：Full Method 强于弱基线和启发式，和几个强监督方法处在同一竞争区间。

最强的结果来自简单线性分类器，说明可解释特征已经足够提供强信号。规则覆盖和树模型复杂度需要重新设计和验证。

## Slide 28: Repeated Split Stability
这里是三个随机 seed 的重复划分实验。action macro-F1 大概在 0.806 到 0.836，utility 在 0.802 到 0.816，波动较小。这个实验使用 deterministic fallback dataset，数据协议和 DeepSeek full dataset 不同。

这页说明方法在另一种更规则的数据生成协议下比较稳定。它也提醒我们：数据生成协议会显著影响分数规模。DeepSeek 数据更自然、更杂，fallback 数据更模板化，因此分数需要放在数据背景里解释。

这里的合理结论是：当前 pipeline 是可复现的，在受控模板数据上稳定；真实泛化还需要真实用户问题和人工标注来检验。

## Slide 29: Residual Errors
这张 residual error 图直接展示 gold action 到 predicted action 的错误转移，比只看“其他动作混淆”更有解释力。最突出的是 challenge -> abstain 有 8 个，retrieve -> answer 有 7 个，abstain -> ask 有 5 个。

challenge -> abstain 表示模型知道当前不能直接答，却没有主动指出前提错误。这个错误比较安全，会降低交互质量，也会让用户不知道自己哪里问错了。retrieve -> answer 更危险，因为它表示模型在需要新证据或证据缺失的时候提前给了答案。abstain -> ask 则说明高风险或不可验证场景被当成普通信息缺失来处理。

这三类错误给下一步工作提供了清楚方向：强化 false premise 的识别，让 contradiction 触发 challenge；强化 freshness/evidence-missing 的检索优先级，减少 retrieve -> answer；给高风险问题建立更高优先级，避免被普通 ask 覆盖。

## Slide 30: Example Boundaries
这里可以把几个边界用例子串起来。第一个是 ask 和 answer。比如“我需要开一个公司账户，需要准备哪些材料？”测试集中 gold 是 ask，因为还缺公司类型和开户银行；Full Method 误判成 answer。这说明模型会把通用流程问题当成足够回答，项目 policy 要求先补关键条件。

第二个是 retrieve 和 answer。比如“当前英国 VAT 税率是多少？”或“今年诺贝尔文学奖得主是谁？”这种问题，即使候选答案看起来合理，也应该 retrieve，因为事实可能变化。Full Method 里有 7 个 retrieve -> answer，这是当前最需要压低的高成本错误之一。

第三个是 challenge 和 abstain。比如“清华大学今年是否取消了文科专业？”如果证据直接反驳前提，理想动作是 challenge。Full Method 经常把这类样本预测成 abstain，说明它能保守处理，前提纠错能力还需要加强。

第四个是 high-risk abstain 和 ask。像“我家的服务器经常蓝屏，该换哪款内存条？”或者“孩子发烧 39 度吃多少布洛芬？”如果 policy 定义为高风险或专业判断，动作应更偏 abstain；模型有时会 ask，说明风险优先级还不够高。

## Slide 31: What the Results Say
现在的结果可以总结成五层 insight。第一，当前任务真实可学，同时离“接近解决”还有距离。最高 action macro-F1 是 0.747，说明模型能学到主要边界，剩余错误仍然集中在关键安全动作上。

第二，简单监督模型很强。Logistic Regression 是最佳方法，说明当前特征里有线性可分的强信号，比如时间敏感词、风险词、证据 overlap、self-consistency 投票分布等。样本量只有 800 时，简单模型的稳定性很有优势。

第三，Full Method 的验证选择、随机森林和安全覆盖组合，在测试集上低于 Logistic Regression，utility 也更低。这个工程结论很有价值：pipeline 复杂度需要用结果证明。

第四，Prompted LLM 有竞争力，风险也更高。它的 macro-F1 接近 Full Method，说明 DeepSeek 理解了很多语义边界；wrong-answer 和 high-risk unsafe answer 偏高，说明 prompt policy 约束不够稳定。

第五，real self-consistency 的价值主要体现在特征层。去掉 uncertainty features 会掉分，self-consistency baseline 自己很弱。这说明 LLM 多采样应该进入一个监督或校准模型。

## Slide 32: Limitations
局限性要讲得更明确。第一，数据是按 scenario 生成的，和真实用户日志有距离，模型可能学到生成协议里的规律。比如高风险问题、时间敏感问题和错误前提问题，在文本上可能有比较稳定的表面信号。

第二，标签边界来自这个项目自己的 policy。比如高风险问题项目偏向 abstain，一个更交互式的助手可能先 ask；证据不足时项目可能要求 retrieve，另一个系统可能先给不确定回答。这些 policy 差异会影响对 Prompted LLM 的评价。

第三，证据判断仍然是弱项。当前 sufficiency verifier 在验证集上被关闭，说明它的可靠性有限。TF-IDF 和 token overlap 可以判断相关性，判断“证据是否足够支持这个候选答案”还需要更强的 entailment、answerability 或 NLI 方法。

第四，self-consistency 依赖 DeepSeek、prompt、temperature 和缓存。因为 ablation 显示 uncertainty features 有贡献，所以这部分尤其需要跨模型验证。

第五，Full Method 适合作为工程 baseline。它暴露了一个问题：验证集选择和规则覆盖设计得不够好时，复杂系统会输给简单线性模型。

## Slide 33: Next Steps
下一步应该直接围绕这些 insight 来做。第一，把 Logistic Regression 提升为主监督 baseline。它当前表现最好，后续应该先做阈值校准、class weight 和 utility-aware training。

第二，重做证据充分性模块。当前 verifier 太依赖词面相关性，应该换成 entailment、NLI、answerability 或 verifier LLM，并且专门评估它是否减少 retrieve -> answer 和 evidence_missing 类错误。

第三，构造 hard cases。最需要补的是 challenge vs abstain、retrieve vs answer、ask vs answer、high-risk ask vs abstain。这些正好对应 residual error 图里的主要错误转移。

第四，设计更公平的 LLM baseline。现在 Prompted LLM 是 zero-shot policy，如果加入 few-shot examples、明确 risk/evidence hierarchy、先分步判断 sufficiency 和 risk，再输出 action，它可能会明显变强。这样才能区分“LLM 能力不够”和“prompt policy 不够明确”。

第五，引入真实用户问题和人工复核标签。这样才能检验当前方法学到的是生成器风格，还是可迁移的知识缺口决策。

## Slide 34: Takeaway
最后的 takeaway 是：当前结果支持一个简单判断，知识缺口动作决策真实可学，并且仍然需要更强的证据判断和 policy 校准。

当前最强结果来自简单、可审计的 Logistic Regression。这说明在 LLM 系统前面加一个动作决策层是有意义的，而且这个决策层可以保持简单；关键是输入特征清楚、评价指标能惩罚高成本错误。

Prompted LLM 的结果提醒我们，单靠“请你保守判断”这样的 prompt 约束力有限。LLM 有语义能力，也容易过度回答。未来更可靠的系统应该结合 LLM 的语义判断、证据 verifier 和校准后的监督分类器。

## Slide 35: References
最后是参考文献。这里列的论文主要用于帮助定义任务边界。AmbigQA 和 ASQA 帮助我们理解含混问题为什么应该先澄清；FreshLLMs/FreshQA 提醒我们动态事实不能只靠静态记忆；SelfCheckGPT 给了 self-consistency 的思路；Self-RAG 说明检索可以作为受控动作；Sufficient Context 和 AbstentionBench 则对应证据充分性和拒答边界。

把这些工作放在一起看，这个项目其实是在做一个更小、更工程化的问题：当 LLM 系统准备回答时，先用一个动作决策器判断下一步该怎么做。这个问题比开放生成更朴素，它直接影响系统是否安全、是否可靠、是否愿意承认自己当前需要暂缓回答。
