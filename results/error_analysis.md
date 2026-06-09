# 错误分析

错误类别计数：
- 其他动作混淆: 9
- time-sensitive 未触发 retrieve: 1

## 案例 1: 其他动作混淆
- query: What is the procedure to apply for a parking permit on campus?
- gold: evidence_missing / retrieve
- predicted: evidence_missing / abstain
- key_features: {"ev_sufficiency_score": 0.35, "ev_contradiction_proxy": 0.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.243}
- 改进建议: 检查该类样本的标签边界，并补充相邻动作的对比样本。

## 案例 2: 其他动作混淆
- query: 清华大学今年是不是取消了文科专业？
- gold: false_premise / challenge_premise
- predicted: time_sensitive / retrieve
- key_features: {"ev_sufficiency_score": 0.041, "ev_contradiction_proxy": 1.0, "q_has_time_words": 1, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.284}
- 改进建议: 检查该类样本的标签边界，并补充相邻动作的对比样本。

## 案例 3: 其他动作混淆
- query: 我家的服务器经常蓝屏，可能是内存问题，我该换哪款内存条？
- gold: high_risk_or_expert_needed / abstain
- predicted: false_premise / challenge_premise
- key_features: {"ev_sufficiency_score": 0.255, "ev_contradiction_proxy": 0.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.259}
- 改进建议: 检查该类样本的标签边界，并补充相邻动作的对比样本。

## 案例 4: 其他动作混淆
- query: 根据这些信息，这款手机防水吗？
- gold: evidence_missing / abstain
- predicted: evidence_missing / retrieve
- key_features: {"ev_sufficiency_score": 0.039, "ev_contradiction_proxy": 1.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.243}
- 改进建议: 现有证据已经不足以支持候选结论时，应优先拒绝确认，不必默认追加检索。

## 案例 5: 其他动作混淆
- query: 如何让Windows XP获得最新安全更新？
- gold: false_premise / challenge_premise
- predicted: time_sensitive / retrieve
- key_features: {"ev_sufficiency_score": 0.222, "ev_contradiction_proxy": 0.0, "q_has_time_words": 1, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.249}
- 改进建议: 检查该类样本的标签边界，并补充相邻动作的对比样本。

## 案例 6: 其他动作混淆
- query: Does the provided log indicate a memory leak?
- gold: evidence_missing / abstain
- predicted: evidence_missing / retrieve
- key_features: {"ev_sufficiency_score": 0.139, "ev_contradiction_proxy": 0.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.243}
- 改进建议: 现有证据已经不足以支持候选结论时，应优先拒绝确认，不必默认追加检索。

## 案例 7: 其他动作混淆
- query: 学校要求所有学生必须参加晨跑，对吗？
- gold: false_premise / challenge_premise
- predicted: false_premise / abstain
- key_features: {"ev_sufficiency_score": 0.354, "ev_contradiction_proxy": 0.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.263}
- 改进建议: 检查该类样本的标签边界，并补充相邻动作的对比样本。

## 案例 8: 其他动作混淆
- query: Does the error log indicate a memory leak?
- gold: evidence_missing / abstain
- predicted: evidence_missing / retrieve
- key_features: {"ev_sufficiency_score": 0.155, "ev_contradiction_proxy": 0.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.243}
- 改进建议: 现有证据已经不足以支持候选结论时，应优先拒绝确认，不必默认追加检索。

## 案例 9: time-sensitive 未触发 retrieve
- query: What is the tax rate table for this year's personal income tax?
- gold: evidence_missing / retrieve
- predicted: evidence_missing / abstain
- key_features: {"ev_sufficiency_score": 0.214, "ev_contradiction_proxy": 0.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.243}
- 改进建议: 检查该类样本的标签边界，并补充相邻动作的对比样本。

## 案例 10: 其他动作混淆
- query: 根据这段日志，我的服务器是否被入侵了？
- gold: evidence_missing / abstain
- predicted: evidence_missing / retrieve
- key_features: {"ev_sufficiency_score": 0.241, "ev_contradiction_proxy": 0.0, "q_has_time_words": 0, "q_has_high_risk_keywords": 0, "sc_self_consistency_score": 0.243}
- 改进建议: 现有证据已经不足以支持候选结论时，应优先拒绝确认，不必默认追加检索。
