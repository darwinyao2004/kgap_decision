from knowledge_gap_decision.evaluate import evaluate_prediction, utility_components


def _records():
    return [
        {"gap_type": "sufficient_information", "gold_action": "answer", "false_premise_flag": False, "risk_level": "low", "evidence_sufficiency_label": "sufficient"},
        {"gap_type": "ambiguous_question", "gold_action": "ask", "false_premise_flag": False, "risk_level": "low", "evidence_sufficiency_label": "insufficient"},
        {"gap_type": "false_premise", "gold_action": "challenge_premise", "false_premise_flag": True, "risk_level": "low", "evidence_sufficiency_label": "contradictory"},
    ]


def test_utility_penalizes_wrong_answer():
    records = _records()
    good = utility_components([r["gold_action"] for r in records], ["answer", "ask", "challenge_premise"], records)
    bad = utility_components([r["gold_action"] for r in records], ["answer", "answer", "answer"], records)
    assert good["score"] > bad["score"]
    assert bad["wrong_answer_rate"] > 0


def test_evaluate_prediction_basic():
    records = _records()
    out = evaluate_prediction(
        "x",
        records,
        [r["gap_type"] for r in records],
        [r["gold_action"] for r in records],
    )
    assert out["action_accuracy"] == 1.0
