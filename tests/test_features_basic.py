from knowledge_gap_decision.data_build import build_dataset
from knowledge_gap_decision.features import compute_features


def test_features_basic_columns():
    records = build_dataset(12)
    df = compute_features(records, cache_samples=False)
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
