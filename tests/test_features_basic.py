from knowledge_gap_decision.data_build import build_dataset
from knowledge_gap_decision.features import compute_features


class FakeLLMClient:
    enabled = True

    def chat_json(self, messages, *, temperature=0.0, max_tokens=1024, cache_key=None):
        prompt = "\n".join(m["content"] for m in messages)
        for forbidden in [
            "gap_type",
            "gold_action",
            "final_answer",
            "gold_clarifying_question",
            "required_slots",
            "risk_level",
            "evidence_sufficiency_label",
        ]:
            assert forbidden not in prompt
        idx = int(str(cache_key).rsplit("_", 1)[-1])
        actions = ["answer", "retrieve", "retrieve", "ask", "abstain"]
        return {"action": actions[idx % len(actions)], "rationale": f"short rationale {idx}"}


def test_features_basic_columns():
    records = build_dataset(12)
    df = compute_features(records, cache_samples=False, llm_client=FakeLLMClient())
    for col in [
        "q_char_len",
        "q_has_time_words",
        "ev_top1_similarity",
        "ev_sufficiency_score",
        "sc_self_consistency_score",
    ]:
        assert col in df.columns
    assert len(df) == len(records)
    assert df["ev_sufficiency_score"].between(0, 1).all()


def test_features_do_not_change_when_gold_fields_change():
    record = build_dataset(12)[0]
    modified = dict(record)
    modified.update(
        {
            "gap_type": "high_risk_or_expert_needed",
            "gold_action": "abstain",
            "final_answer": "changed gold answer",
            "gold_clarifying_question": "changed gold question",
            "required_slots": ["changed_slot"],
            "risk_level": "high",
            "evidence_sufficiency_label": "contradictory",
        }
    )
    df = compute_features([record, modified], cache_samples=False, llm_client=FakeLLMClient())
    feature_cols = [c for c in df.columns if c != "id"]
    assert df.loc[0, feature_cols].to_dict() == df.loc[1, feature_cols].to_dict()
