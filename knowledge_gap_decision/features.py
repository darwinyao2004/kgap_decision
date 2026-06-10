import json
import math
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import variance
from typing import Any

import numpy as np
import pandas as pd

from .retrieval import tfidf_scores, token_overlap, tokenize
from .schema import ACTION_TYPES, Action

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
INPUT_FIELDS = ("user_initial_query", "dialogue_context", "retrieved_evidence", "candidate_answer")
SUPERVISED_FIELDS = {
    "gap_type",
    "gold_action",
    "final_answer",
    "gold_clarifying_question",
    "required_slots",
    "evidence_sufficiency_label",
    "false_premise_flag",
    "time_sensitive_flag",
    "risk_level",
}
SC_CACHE_SCHEMA = "llm_self_consistency_v1"


class SelfConsistencySamplingError(RuntimeError):
    pass


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


def _input_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_initial_query": row["user_initial_query"],
        "dialogue_context": row["dialogue_context"],
        "retrieved_evidence": row["retrieved_evidence"],
        "candidate_answer": row["candidate_answer"],
    }


def _input_hash(row: dict[str, Any]) -> str:
    text = json.dumps(_input_payload(row), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_sc_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not cache_path.exists():
        return cache
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("schema") != SC_CACHE_SCHEMA:
            continue
        if any(field in item for field in SUPERVISED_FIELDS):
            continue
        key = item.get("input_hash")
        outputs = item.get("outputs")
        if isinstance(key, str) and isinstance(outputs, list):
            cache[key] = item
    return cache


def _normalize_action(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "clarify": Action.ASK.value,
        "ask_user": Action.ASK.value,
        "search": Action.RETRIEVE.value,
        "look_up": Action.RETRIEVE.value,
        "refuse": Action.ABSTAIN.value,
        "reject": Action.ABSTAIN.value,
        "challenge": Action.CHALLENGE.value,
        "correct_premise": Action.CHALLENGE.value,
    }
    return normalized if normalized in ACTION_TYPES else aliases.get(normalized)


def _sample_action_decisions(
    row: dict[str, Any],
    client: Any,
    *,
    sample_count: int,
    temperature: float,
) -> list[dict[str, str]]:
    if client is None or not getattr(client, "enabled", False):
        raise SelfConsistencySamplingError("LLM self-consistency requires DEEPSEEK_API_KEY; no enabled client was provided.")
    payload = _input_payload(row)
    system = (
        "You are a conservative QA decision policy sampler. Use only the four input fields in the user payload. "
        "Do not rely on hidden labels or any supervision fields. Return strict JSON only. "
        f"`action` must be one of: {', '.join(ACTION_TYPES)}. "
        "Use `answer` only when the candidate answer is sufficiently supported; "
        "use `ask` when user constraints or ambiguity are missing; "
        "use `retrieve` when evidence is missing or freshness is required; "
        "use `abstain` when the system should not confirm the candidate answer; "
        "use `challenge_premise` when the query premise is unsupported or contradicted."
    )
    outputs: list[dict[str, str]] = []
    base_key = _input_hash(row)
    for i in range(sample_count):
        result = client.chat_json(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        "Return JSON with exactly these keys: `action` and `rationale`. "
                        "The rationale must be short, one sentence, and must not mention hidden labels.\n"
                        + json.dumps(payload, ensure_ascii=False, indent=2)
                    ),
                },
            ],
            temperature=temperature,
            max_tokens=2048,
            cache_key=f"self_consistency_{base_key}_{i}",
        )
        if not isinstance(result, dict):
            raise SelfConsistencySamplingError(f"LLM self-consistency call failed for record {row.get('id', '<unknown>')}.")
        action = _normalize_action(result.get("action"))
        rationale = result.get("rationale")
        if action not in ACTION_TYPES or not isinstance(rationale, str) or not rationale.strip():
            raise SelfConsistencySamplingError(f"Malformed LLM self-consistency output for record {row.get('id', '<unknown>')}.")
        outputs.append({"action": action, "rationale": rationale.strip()})
    return outputs


def _cache_item(row: dict[str, Any], outputs: list[dict[str, str]], sample_count: int, temperature: float) -> dict[str, Any]:
    return {
        "schema": SC_CACHE_SCHEMA,
        "id": row.get("id", ""),
        "input_hash": _input_hash(row),
        "sample_count": sample_count,
        "temperature": temperature,
        "input_fields": list(INPUT_FIELDS),
        "outputs": outputs,
    }


def _append_cache_item(cache_path: Path, item: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _ensure_llm_self_consistency_cache(
    records: list[dict[str, Any]],
    *,
    client: Any,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    cache_samples: bool,
    sample_count: int,
    temperature: float,
    max_workers: int,
) -> None:
    missing: dict[str, dict[str, Any]] = {}
    for row in records:
        key = _input_hash(row)
        cached = cache.get(key)
        if cached and len(cached.get("outputs", [])) >= sample_count:
            continue
        missing.setdefault(key, row)
    if not missing:
        return
    workers = max(1, max_workers)
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {
        executor.submit(_sample_action_decisions, row, client, sample_count=sample_count, temperature=temperature): row
        for row in missing.values()
    }
    try:
        for future in as_completed(futures):
            row = futures[future]
            outputs = future.result()
            item = _cache_item(row, outputs, sample_count, temperature)
            if cache_samples:
                _append_cache_item(cache_path, item)
            cache[item["input_hash"]] = item
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


def llm_self_consistency_outputs(
    row: dict[str, Any],
    *,
    client: Any,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    cache_samples: bool,
    sample_count: int = 5,
    temperature: float = 0.7,
) -> list[dict[str, str]]:
    key = _input_hash(row)
    cached = cache.get(key)
    if cached and len(cached.get("outputs", [])) >= sample_count:
        return cached["outputs"][:sample_count]

    outputs = _sample_action_decisions(row, client, sample_count=sample_count, temperature=temperature)
    item = _cache_item(row, outputs, sample_count, temperature)
    if cache_samples:
        _append_cache_item(cache_path, item)
    cache[key] = item
    return outputs


def self_consistency_features(row: dict[str, Any], outputs: list[dict[str, str]]) -> dict[str, float]:
    rationales = [o["rationale"] for o in outputs]
    actions = [o["action"] for o in outputs]
    lengths = [len(s) for s in rationales]
    ents = [_entity_count(s) for s in rationales]
    dates = [len(_dates(s)) for s in rationales]
    nums = [len(_numbers(s)) for s in rationales]
    normalized = [re.sub(r"\s+", "", s.lower()) for s in rationales]
    majority = max(normalized.count(s) for s in set(normalized)) / max(1, len(normalized))
    pair_overlaps = []
    for i, a in enumerate(rationales):
        for b in rationales[i + 1 :]:
            pair_overlaps.append(token_overlap(a, b))
    rationale_overlap = float(np.mean(pair_overlaps)) if pair_overlaps else 1.0
    action_counts = {action: actions.count(action) for action in ACTION_TYPES}
    n = max(1, len(actions))
    probs = [count / n for count in action_counts.values() if count]
    entropy = -sum(p * math.log(p) for p in probs) / math.log(len(ACTION_TYPES)) if probs else 0.0
    majority_action_ratio = max(action_counts.values()) / n
    answer_votes = action_counts[Action.ANSWER.value]
    non_answer_votes = n - answer_votes
    answer_reject_disagreement = 1.0 - abs(answer_votes - non_answer_votes) / n
    return {
        "sc_action_vote_entropy": float(entropy),
        "sc_majority_action_ratio": float(majority_action_ratio),
        "sc_answer_vote_count": float(answer_votes),
        "sc_ask_vote_count": float(action_counts[Action.ASK.value]),
        "sc_retrieve_vote_count": float(action_counts[Action.RETRIEVE.value]),
        "sc_abstain_vote_count": float(action_counts[Action.ABSTAIN.value]),
        "sc_challenge_premise_vote_count": float(action_counts[Action.CHALLENGE.value]),
        "sc_answer_reject_disagreement": float(answer_reject_disagreement),
        "sc_rationale_text_overlap": rationale_overlap,
        "sc_rationale_text_disagreement": 1.0 - rationale_overlap,
        "sc_answer_embedding_disagreement": 1.0 - rationale_overlap,
        "sc_answer_length_variance": float(variance(lengths)) if len(lengths) > 1 else 0.0,
        "sc_entity_disagreement": float(np.std(ents)),
        "sc_date_disagreement": float(np.std(dates)),
        "sc_number_disagreement": float(np.std(nums)),
        "sc_majority_answer_ratio": float(majority_action_ratio),
        "sc_self_consistency_score": float(0.60 * majority_action_ratio + 0.40 * rationale_overlap),
    }


def compute_features(
    records: list[dict[str, Any]],
    cache_samples: bool = True,
    *,
    llm_client: Any | None = None,
    sc_cache_path: str | Path = "data/cache/llm_self_consistency.jsonl",
    sc_sample_count: int = 5,
    sc_temperature: float = 0.7,
    sc_max_workers: int = 20,
) -> pd.DataFrame:
    cache_path = Path(sc_cache_path)
    sc_cache = _load_sc_cache(cache_path)
    _ensure_llm_self_consistency_cache(
        records,
        client=llm_client,
        cache=sc_cache,
        cache_path=cache_path,
        cache_samples=cache_samples,
        sample_count=sc_sample_count,
        temperature=sc_temperature,
        max_workers=sc_max_workers,
    )
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
        sc_outputs = llm_self_consistency_outputs(
            row,
            client=llm_client,
            cache=sc_cache,
            cache_path=cache_path,
            cache_samples=cache_samples,
            sample_count=sc_sample_count,
            temperature=sc_temperature,
        )
        sc = self_consistency_features(row, sc_outputs)
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
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame, *, include_evidence: bool = True, include_uncertainty: bool = True) -> list[str]:
    cols = [c for c in df.columns if c != "id"]
    if not include_evidence:
        cols = [c for c in cols if not c.startswith("ev_")]
    if not include_uncertainty:
        cols = [c for c in cols if not c.startswith("sc_")]
    return cols
