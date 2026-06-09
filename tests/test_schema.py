from knowledge_gap_decision.data_build import build_dataset
from knowledge_gap_decision.schema import ACTION_TYPES, GAP_TYPES, validate_record


def test_generated_records_validate():
    records = build_dataset(20)
    assert records
    for record in records:
        assert validate_record(record) == []
        assert record["gap_type"] in GAP_TYPES
        assert record["gold_action"] in ACTION_TYPES
