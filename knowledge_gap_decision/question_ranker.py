import re
from typing import Any

import pandas as pd


def generate_candidates(record: dict[str, Any]) -> list[dict[str, str]]:
    slots = record.get("required_slots") or ["限定条件"]
    query = record["user_initial_query"]
    candidates = []
    for slot in slots[:4]:
        candidates.append(
            {
                "question": f"请补充{slot}，这样我才能更准确地回答你的问题。",
                "target_slot": slot,
                "why_needed": f"当前问题缺少{slot}。",
            }
        )
    if len(slots) > 1:
        candidates.append(
            {
                "question": f"你的{slots[0]}和{slots[1]}分别是什么？",
                "target_slot": ",".join(slots[:2]),
                "why_needed": "这些条件会改变推荐或解答路径。",
            }
        )
    candidates.append(
        {
            "question": "你希望我按哪一种具体场景来回答？",
            "target_slot": "场景",
            "why_needed": f"原问题“{query[:24]}”存在多种解释。",
        }
    )
    return candidates[:5]


def _is_leading(question: str) -> bool:
    return bool(re.search(r"是不是|是否就是|你其实|对吧", question))


def score_question(candidate: dict[str, str], record: dict[str, Any]) -> float:
    q = candidate["question"]
    target = candidate.get("target_slot", "")
    slots = record.get("required_slots") or []
    covered = sum(1 for s in slots if s and s in target + q)
    slot_coverage = covered / max(1, len(slots))
    one_core = 1.0 if q.count("？") + q.count("?") <= 1 and len(re.split("和|、|,|，", target)) <= 2 else 0.65
    specificity = min(1.0, (len(target) + 4) / 12)
    answerability = 1.0 if len(q) <= 45 else 0.75
    politeness = 1.0 if q.startswith("请") or "可以" in q else 0.85
    uncertainty_reduction = 0.8 if covered else 0.45
    penalty = 0.25 if _is_leading(q) else 0.0
    return round(
        0.45 * slot_coverage
        + 0.25 * uncertainty_reduction
        + 0.15 * specificity
        + 0.10 * answerability
        + 0.05 * politeness
        - penalty,
        4,
    )


def rank_questions(records: list[dict[str, Any]], action_pred: list[str]) -> pd.DataFrame:
    rows = []
    for record, pred in zip(records, action_pred):
        if record["gold_action"] != "ask" and pred != "ask":
            continue
        candidates = generate_candidates(record)
        scored = [(c, score_question(c, record)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        best, score = scored[0]
        rows.append(
            {
                "id": record["id"],
                "gold_action": record["gold_action"],
                "pred_action": pred,
                "required_slots": "|".join(record.get("required_slots", [])),
                "best_clarifying_question": best["question"],
                "target_slot": best["target_slot"],
                "question_utility_score": score,
                "candidate_count": len(candidates),
                "covers_required_slot": int(score >= 0.55),
                "only_one_core_question": int(best["question"].count("？") + best["question"].count("?") <= 1),
                "non_leading": int(not _is_leading(best["question"])),
            }
        )
    return pd.DataFrame(rows)
