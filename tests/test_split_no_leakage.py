from knowledge_gap_decision.data_build import assert_no_group_leakage, build_dataset, split_by_group


def test_split_has_no_group_leakage():
    records = build_dataset(100)
    splits = split_by_group(records)
    assert_no_group_leakage(splits)
    groups_by_split = {k: {r["group_id"] for r in v} for k, v in splits.items()}
    assert groups_by_split["train"].isdisjoint(groups_by_split["val"])
    assert groups_by_split["train"].isdisjoint(groups_by_split["test"])
    assert groups_by_split["val"].isdisjoint(groups_by_split["test"])
