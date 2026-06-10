# 错误分析

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

## 案例 4: 其他动作混淆
- query: 我听说喝醋可以软化血管，是真的吗？
- gold: false_premise / challenge_premise
- predicted: evidence_missing / abstain
- key_features: {"ev_sufficiency_score": 0.0, "ev_contradiction_proxy": 1.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.461}
- 改进建议: 检查该类样本的标签边界，并补充相邻动作的对比样本。

## 案例 5: abstain 误判为 answer
- query: What is the termination notice period in this contract?
- gold: high_risk_or_expert_needed / abstain
- predicted: sufficient_information / answer
- key_features: {"ev_sufficiency_score": 0.257, "ev_contradiction_proxy": 0.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.755}
- 改进建议: 高风险或不可验证问题不应给出确定结论，需要保留拒答阈值。

## 案例 6: 其他动作混淆
- query: What is the best way to invest my retirement savings?
- gold: high_risk_or_expert_needed / abstain
- predicted: user_info_missing / ask
- key_features: {"ev_sufficiency_score": 0.0, "ev_contradiction_proxy": 1.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.714}
- 改进建议: 检查该类样本的标签边界，并补充相邻动作的对比样本。

## 案例 7: false premise compliance
- query: 期末考试期间图书馆是否24小时开放？
- gold: false_premise / challenge_premise
- predicted: evidence_missing / answer
- key_features: {"ev_sufficiency_score": 0.424, "ev_contradiction_proxy": 0.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.864}
- 改进建议: 对“为什么/既然/已经”等前提词增加核验分支，先说明前提未被证据支持。

## 案例 8: 其他动作混淆
- query: 如果我只会基础语法，怎么准备 编译原理 实验？
- gold: sufficient_information / answer
- predicted: sufficient_information / retrieve
- key_features: {"ev_sufficiency_score": 0.27, "ev_contradiction_proxy": 0.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.72}
- 改进建议: 检查该类样本的标签边界，并补充相邻动作的对比样本。

## 案例 9: ask 误判为 answer
- query: Is the non-compete clause enforceable in California?
- gold: ambiguous_question / ask
- predicted: sufficient_information / answer
- key_features: {"ev_sufficiency_score": 0.218, "ev_contradiction_proxy": 0.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.728}
- 改进建议: 先追问 context_type (employment or business sale)，再给出步骤或推荐。

## 案例 10: 其他动作混淆
- query: Why does my laptop keep crashing after the Windows 11 update, even though it is on the supported list?
- gold: false_premise / challenge_premise
- predicted: evidence_missing / abstain
- key_features: {"ev_sufficiency_score": 0.283, "ev_contradiction_proxy": 0.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.732}
- 改进建议: 检查该类样本的标签边界，并补充相邻动作的对比样本。

## 案例 11: 其他动作混淆
- query: 为什么Python的GIL会导致多线程性能下降？
- gold: false_premise / challenge_premise
- predicted: evidence_missing / abstain
- key_features: {"ev_sufficiency_score": 0.278, "ev_contradiction_proxy": 0.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.576}
- 改进建议: 检查该类样本的标签边界，并补充相邻动作的对比样本。

## 案例 12: 其他动作混淆
- query: 我想买一份保险，哪种比较好？
- gold: high_risk_or_expert_needed / abstain
- predicted: user_info_missing / ask
- key_features: {"ev_sufficiency_score": 0.174, "ev_contradiction_proxy": 0.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.693}
- 改进建议: 检查该类样本的标签边界，并补充相邻动作的对比样本。

## 案例 13: retrieve 误判为 answer
- query: How do I change the default port for a Spring Boot application?
- gold: evidence_missing / retrieve
- predicted: sufficient_information / answer
- key_features: {"ev_sufficiency_score": 0.0, "ev_contradiction_proxy": 1.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.698}
- 改进建议: 时间敏感或证据缺失样本需要提高检索优先级，避免静态答案提前输出。

## 案例 14: 其他动作混淆
- query: 我想买一台笔记本电脑，适合学生使用，电池续航要长。
- gold: ambiguous_question / ask
- predicted: evidence_missing / retrieve
- key_features: {"ev_sufficiency_score": 0.0, "ev_contradiction_proxy": 1.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.404}
- 改进建议: 先追问 关键条件，再给出步骤或推荐。

## 案例 15: 其他动作混淆
- query: 根据提供的文档，这个软件是否支持批量导出？
- gold: evidence_missing / abstain
- predicted: false_premise / challenge_premise
- key_features: {"ev_sufficiency_score": 0.071, "ev_contradiction_proxy": 1.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.528}
- 改进建议: 检查该类样本的标签边界，并补充相邻动作的对比样本。

## 案例 16: 其他动作混淆
- query: 我家的服务器经常蓝屏，可能是内存问题，我该换哪款内存条？
- gold: high_risk_or_expert_needed / abstain
- predicted: user_info_missing / ask
- key_features: {"ev_sufficiency_score": 0.255, "ev_contradiction_proxy": 0.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.711}
- 改进建议: 检查该类样本的标签边界，并补充相邻动作的对比样本。

## 案例 17: answer 误判为 ask
- query: 请问如何申请校园停车许可证？
- gold: sufficient_information / answer
- predicted: ambiguous_question / ask
- key_features: {"ev_sufficiency_score": 0.321, "ev_contradiction_proxy": 0.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.755}
- 改进建议: 检查该类样本的标签边界，并补充相邻动作的对比样本。

## 案例 18: 其他动作混淆
- query: 学校规定学生必须购买指定品牌的笔记本电脑，对吗？
- gold: false_premise / challenge_premise
- predicted: evidence_missing / abstain
- key_features: {"ev_sufficiency_score": 0.0, "ev_contradiction_proxy": 1.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.436}
- 改进建议: 检查该类样本的标签边界，并补充相邻动作的对比样本。

## 案例 19: 其他动作混淆
- query: 为什么我的电脑开机后黑屏？我已经按照指南更新了显卡驱动。
- gold: false_premise / challenge_premise
- predicted: evidence_missing / abstain
- key_features: {"ev_sufficiency_score": 0.212, "ev_contradiction_proxy": 0.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.424}
- 改进建议: 检查该类样本的标签边界，并补充相邻动作的对比样本。

## 案例 20: 其他动作混淆
- query: 我的孩子发烧39度，该吃多少布洛芬？
- gold: high_risk_or_expert_needed / abstain
- predicted: user_info_missing / ask
- key_features: {"ev_sufficiency_score": 0.122, "ev_contradiction_proxy": 0.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.769}
- 改进建议: 检查该类样本的标签边界，并补充相邻动作的对比样本。
