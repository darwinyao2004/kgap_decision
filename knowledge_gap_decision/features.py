import json
import re
from pathlib import Path
from statistics import variance
from typing import Any

import numpy as np
import pandas as pd

from .retrieval import tfidf_scores, token_overlap, tokenize

TIME_WORDS = {"today", "current", "latest", "now", "recent", "今年", "现在", "最近", "最新", "今天", "目前", "截止"}
SUBJECTIVE_WORDS = {"best", "should", "recommend", "worth", "好不好", "是否值得", "推荐", "应该", "最好"}
CONDITION_WORDS = {"if", "when", "depending", "depends", "如果", "取决于", "假如", "条件"}
SUPPORT_JUDGMENT_WORDS = {"证明", "支持", "能不能", "能否", "是否能", "whether", "support", "prove"}
RISK_WORDS = {
    "medical",
    "legal",
    "financial",
    "diagnosis",
    "treatment",
    "lawsuit",
    "investment",
    "医学",
    "诊断",
    "法律",
    "投资",
    "处方",
    "胸痛",
    "合同",
    "药",
}
EN_NEG_WORDS = {"not", "no", "never"}
ZH_NEG_WORDS = {"没有", "并未", "不能", "不是", "未"}


def _contains_any(text: str, words: set[str]) -> int:
    lower = text.lower()
    return int(any(w.lower() in lower for w in words))


def _entity_count(text: str) -> int:
    english_entities = re.findall(r"\b[A-Z][A-Za-z0-9_.-]{1,}\b", text)
    dates = re.findall(r"\b20\d{2}\b|\d{1,2}月\d{1,2}日|\d{4}年", text)
    paths = re.findall(r"/[A-Za-z0-9_./-]+", text)
    chinese_names = re.findall(r"[\u4e00-\u9fff]{2,}(?:大学|系统|银行|公司|网络|课程)", text)
    return len(set(english_entities + dates + paths + chinese_names))


def _false_premise_pattern(text: str) -> int:
    patterns = ["为什么", "既然", "已经", "why is", "why has", "since"]
    lower = text.lower()
    return int(("为什么" in text and "已经" in text) or ("既然" in text) or any(p in lower for p in patterns[3:]))


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def _dates(text: str) -> set[str]:
    return set(re.findall(r"20\d{2}|\d{1,2}月\d{1,2}日|\d{4}年", text))


def contradiction_proxy(query: str, answer: str, evidence: list[str]) -> float:
    joined = " ".join(evidence)
    if not joined:
        return 0.0
    joined_lower = joined.lower()
    answer_lower = answer.lower()
    ev_tokens = set(re.findall(r"\b[a-z]+\b", joined_lower))
    ans_tokens = set(re.findall(r"\b[a-z]+\b", answer_lower))
    neg_ev = bool(ev_tokens & EN_NEG_WORDS) or any(w in joined for w in ZH_NEG_WORDS)
    affirmative_answer = not (bool(ans_tokens & EN_NEG_WORDS) or any(w in answer for w in ZH_NEG_WORDS))
    number_conflict = bool(_numbers(answer) and _numbers(joined) and _numbers(answer).isdisjoint(_numbers(joined)))
    date_conflict = bool(_dates(query) and _dates(joined) and _dates(query).isdisjoint(_dates(joined)))
    return float((neg_ev and affirmative_answer) or number_conflict or date_conflict)


def evidence_sufficiency_score(top1: float, query_overlap: float, answer_overlap: float, contradiction: float, evidence_count: int) -> float:
    # This intentionally avoids annotation fields such as evidence_sufficiency_label.
    count_bonus = min(evidence_count, 3) / 3
    raw = 0.38 * top1 + 0.22 * query_overlap + 0.30 * answer_overlap + 0.10 * count_bonus - 0.30 * contradiction
    return float(np.clip(raw, 0.0, 1.0))


def pseudo_samples(row: dict[str, Any]) -> list[str]:
    base = row.get("candidate_answer", "")
    gap = row.get("gap_type", "")
    q = row.get("user_initial_query", "")
    if gap == "sufficient_information":
        return [base, base.replace("。", ""), base, row.get("final_answer", base), base]
    if gap in {"ambiguous_question", "user_info_missing"}:
        return [
            base,
            row.get("gold_clarifying_question", "需要补充条件。"),
            "需要先确认用户条件。",
            f"这个问题取决于 {', '.join(row.get('required_slots', []) or ['具体条件'])}。",
            "无法在缺少限定条件时直接回答。",
        ]
    if gap == "false_premise":
        return [base, row.get("final_answer", base), "前提可能不成立。", "需要先核验前提。", base]
    if gap == "time_sensitive":
        return [base, "需要检索最新来源。", "当前信息可能已变化。", base.replace("一直", "可能"), "应查询官方资料。"]
    if gap == "high_risk_or_expert_needed":
        return [base, "建议咨询专业人员。", "不能给出专业判断。", "需要完整材料和专家评估。", base]
    return [base, "证据不足。", "需要检索更多资料。", base, "不能确认。"]


def self_consistency_features(row: dict[str, Any], samples: list[str] | None = None) -> dict[str, float]:
    samples = samples or pseudo_samples(row)
    lengths = [len(s) for s in samples]
    ents = [_entity_count(s) for s in samples]
    dates = [len(_dates(s)) for s in samples]
    nums = [len(_numbers(s)) for s in samples]
    normalized = [re.sub(r"\s+", "", s.lower()) for s in samples]
    majority = max(normalized.count(s) for s in set(normalized)) / max(1, len(normalized))
    pair_overlaps = []
    for i, a in enumerate(samples):
        for b in samples[i + 1 :]:
            pair_overlaps.append(token_overlap(a, b))
    agreement = float(np.mean(pair_overlaps)) if pair_overlaps else 1.0
    return {
        "sc_answer_embedding_disagreement": 1.0 - agreement,
        "sc_answer_length_variance": float(variance(lengths)) if len(lengths) > 1 else 0.0,
        "sc_entity_disagreement": float(np.std(ents)),
        "sc_date_disagreement": float(np.std(dates)),
        "sc_number_disagreement": float(np.std(nums)),
        "sc_majority_answer_ratio": float(majority),
        "sc_self_consistency_score": float(0.55 * agreement + 0.45 * majority),
    }


def compute_features(records: list[dict[str, Any]], cache_samples: bool = True) -> pd.DataFrame:
    cache_path = Path("data/cache/llm_samples.jsonl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sample_lines = []
    rows = []
    for row in records:
        q = row["user_initial_query"]
        ctx = row["dialogue_context"]
        evidence = row["retrieved_evidence"]
        answer = row["candidate_answer"]
        joined_ev = " ".join(evidence)
        retrieval = tfidf_scores(q, evidence)
        q_ev_overlap = token_overlap(q, joined_ev)
        ans_ev_overlap = token_overlap(answer, joined_ev)
        contradiction = contradiction_proxy(q, answer, evidence)
        suff_score = evidence_sufficiency_score(retrieval.top1_similarity, q_ev_overlap, ans_ev_overlap, contradiction, len(evidence))
        samples = pseudo_samples(row)
        sc = self_consistency_features(row, samples)
        if cache_samples:
            sample_lines.append(json.dumps({"id": row["id"], "mode": "offline_fallback", "samples": samples}, ensure_ascii=False))
        q_tokens = tokenize(q)
        ev_tokens = tokenize(joined_ev)
        coverage = len(set(q_tokens) & set(ev_tokens)) / max(1, len(set(q_tokens)))
        feature = {
            "id": row["id"],
            "q_char_len": len(q),
            "q_word_count": len(q_tokens),
            "q_has_time_words": _contains_any(q, TIME_WORDS),
            "q_has_subjective_words": _contains_any(q, SUBJECTIVE_WORDS),
            "q_has_condition_words": _contains_any(q, CONDITION_WORDS),
            "q_asks_support_judgment": _contains_any(q, SUPPORT_JUDGMENT_WORDS),
            "q_entity_count": _entity_count(q),
            "q_has_false_premise_pattern": _false_premise_pattern(q),
            "q_has_high_risk_keywords": _contains_any(q, RISK_WORDS),
            "ev_top1_similarity": retrieval.top1_similarity,
            "ev_top5_avg_similarity": retrieval.top5_avg_similarity,
            "ev_query_token_overlap": q_ev_overlap,
            "ev_answer_token_overlap": ans_ev_overlap,
            "ev_coverage_ratio": coverage,
            "ev_contradiction_proxy": contradiction,
            "ev_sufficiency_score": suff_score,
            "ev_count": len(evidence),
            "ev_total_length": len(joined_ev),
            "risk_high": _contains_any(" ".join([q, ctx, answer]), RISK_WORDS),
            "risk_medium": int(_contains_any(" ".join(evidence), {"policy", "contract", "deadline", "合规", "合同", "截止"}) and not _contains_any(" ".join([q, ctx, answer]), RISK_WORDS)),
        }
        feature.update(sc)
        rows.append(feature)
    if cache_samples:
        cache_path.write_text("\n".join(sample_lines) + "\n", encoding="utf-8")
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame, *, include_evidence: bool = True, include_uncertainty: bool = True) -> list[str]:
    cols = [c for c in df.columns if c != "id"]
    if not include_evidence:
        cols = [c for c in cols if not c.startswith("ev_")]
    if not include_uncertainty:
        cols = [c for c in cols if not c.startswith("sc_")]
    return cols
