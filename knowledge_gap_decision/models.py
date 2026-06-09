import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .schema import ACTION_TYPES, GAP_TYPES, Action, GapType, action_from_gap

SEED = 42


@dataclass
class PredictionResult:
    method: str
    gap_pred: list[str]
    action_pred: list[str]
    status: str = "ok"
    explanations: list[str] | None = None


def _majority(values: list[str]) -> str:
    return pd.Series(values).mode().iloc[0]


def always_baseline(test_records: list[dict[str, Any]], method: str, action: str, train_records: list[dict[str, Any]]) -> PredictionResult:
    gap = GapType.SUFFICIENT.value if action == Action.ANSWER.value else _majority([r["gap_type"] for r in train_records])
    return PredictionResult(method, [gap] * len(test_records), [action] * len(test_records))


def rag_threshold(records: list[dict[str, Any]], features: pd.DataFrame) -> PredictionResult:
    gaps, actions, explanations = [], [], []
    for row, (_, f) in zip(records, features.iterrows()):
        if f["q_has_false_premise_pattern"] or f["ev_contradiction_proxy"] > 0.55:
            gap, action, reason = GapType.FALSE_PREMISE.value, Action.CHALLENGE.value, "contradictory evidence or false-premise cue"
        elif f["risk_high"]:
            gap, action, reason = GapType.HIGH_RISK.value, Action.ABSTAIN.value, "high-risk keyword"
        elif f["q_has_time_words"]:
            gap, action, reason = GapType.TIME_SENSITIVE.value, Action.RETRIEVE.value, "time-sensitive cue"
        elif f["ev_sufficiency_score"] > 0.58 and f["ev_top1_similarity"] > 0.08:
            gap, action, reason = GapType.SUFFICIENT.value, Action.ANSWER.value, "evidence similarity above threshold"
        elif f["q_has_subjective_words"] or f["q_has_condition_words"]:
            gap, action, reason = GapType.USER_INFO_MISSING.value, Action.ASK.value, "preference or condition missing"
        else:
            gap, action, reason = GapType.EVIDENCE_MISSING.value, Action.RETRIEVE.value, "low evidence sufficiency"
        gaps.append(gap)
        actions.append(action)
        explanations.append(reason)
    return PredictionResult("RAG Similarity Threshold", gaps, actions, explanations=explanations)


def self_consistency_baseline(records: list[dict[str, Any]], features: pd.DataFrame) -> PredictionResult:
    gaps, actions, explanations = [], [], []
    for row, (_, f) in zip(records, features.iterrows()):
        if f["q_has_false_premise_pattern"]:
            gap, action, reason = GapType.FALSE_PREMISE.value, Action.CHALLENGE.value, "false-premise pattern"
        elif f["risk_high"]:
            gap, action, reason = GapType.HIGH_RISK.value, Action.ABSTAIN.value, "high-risk domain"
        elif f["q_has_time_words"]:
            gap, action, reason = GapType.TIME_SENSITIVE.value, Action.RETRIEVE.value, "freshness required"
        elif f["sc_self_consistency_score"] < 0.45 and (f["q_has_subjective_words"] or f["q_has_condition_words"]):
            gap, action, reason = GapType.AMBIGUOUS.value, Action.ASK.value, "low self-consistency plus ambiguity"
        elif f["sc_self_consistency_score"] > 0.7 and f["ev_sufficiency_score"] > 0.5:
            gap, action, reason = GapType.SUFFICIENT.value, Action.ANSWER.value, "consistent sampled answers"
        else:
            gap, action, reason = GapType.EVIDENCE_MISSING.value, Action.RETRIEVE.value, "uncertain sampled answers"
        gaps.append(gap)
        actions.append(action)
        explanations.append(reason)
    return PredictionResult("Self-Consistency Baseline", gaps, actions, explanations=explanations)


class DualClassifier:
    def __init__(self, model_kind: str, feature_cols: list[str]) -> None:
        self.model_kind = model_kind
        self.feature_cols = feature_cols
        if model_kind == "logistic":
            gap_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
            action_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
            self.gap_model = Pipeline([("scaler", StandardScaler()), ("clf", gap_model)])
            self.action_model = Pipeline([("scaler", StandardScaler()), ("clf", action_model)])
        elif model_kind == "rf":
            self.gap_model = RandomForestClassifier(n_estimators=160, class_weight="balanced", random_state=SEED)
            self.action_model = RandomForestClassifier(n_estimators=160, class_weight="balanced", random_state=SEED)
        elif model_kind == "gbdt":
            self.gap_model = GradientBoostingClassifier(random_state=SEED)
            self.action_model = GradientBoostingClassifier(random_state=SEED)
        else:
            raise ValueError(model_kind)

    def fit(self, features: pd.DataFrame, records: list[dict[str, Any]]) -> "DualClassifier":
        x = features[self.feature_cols]
        self.gap_model.fit(x, [r["gap_type"] for r in records])
        self.action_model.fit(x, [r["gold_action"] for r in records])
        return self

    def predict(self, features: pd.DataFrame, method_name: str) -> PredictionResult:
        x = features[self.feature_cols]
        gaps = self.gap_model.predict(x).tolist()
        actions = self.action_model.predict(x).tolist()
        return PredictionResult(method_name, gaps, actions)


class TextClassifier:
    def __init__(self) -> None:
        self.gap_model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=6000)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
            ]
        )
        self.action_model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=6000)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
            ]
        )

    @staticmethod
    def texts(records: list[dict[str, Any]]) -> list[str]:
        return [
            " [CTX] ".join(
                [
                    r["user_initial_query"],
                    r["dialogue_context"],
                    " ".join(r["retrieved_evidence"]),
                    r["candidate_answer"],
                ]
            )
            for r in records
        ]

    def fit(self, records: list[dict[str, Any]]) -> "TextClassifier":
        texts = self.texts(records)
        self.gap_model.fit(texts, [r["gap_type"] for r in records])
        self.action_model.fit(texts, [r["gold_action"] for r in records])
        return self

    def predict(self, records: list[dict[str, Any]]) -> PredictionResult:
        texts = self.texts(records)
        return PredictionResult(
            "Text Encoder Classifier",
            self.gap_model.predict(texts).tolist(),
            self.action_model.predict(texts).tolist(),
        )


def full_method_predict(
    train_features: pd.DataFrame,
    train_records: list[dict[str, Any]],
    test_features: pd.DataFrame,
    *,
    include_evidence: bool = True,
    include_uncertainty: bool = True,
    use_gap_classifier: bool = True,
    use_evidence_verifier: bool = True,
    use_question_ranker: bool = True,
    method_name: str = "Full Method",
) -> PredictionResult:
    feature_cols = [c for c in train_features.columns if c != "id"]
    if not include_evidence:
        feature_cols = [c for c in feature_cols if not c.startswith("ev_")]
    if not include_uncertainty:
        feature_cols = [c for c in feature_cols if not c.startswith("sc_")]
    clf = DualClassifier("rf", feature_cols).fit(train_features, train_records)
    base = clf.predict(test_features, method_name)
    gaps, actions, explanations = [], [], []
    for i, (_, f) in enumerate(test_features.iterrows()):
        gap = base.gap_pred[i] if use_gap_classifier else ""
        action = base.action_pred[i]
        reasons = []
        if include_evidence and f.get("ev_contradiction_proxy", 0) > 0.65 and (
            f.get("q_has_false_premise_pattern", 0) or f.get("q_asks_support_judgment", 0)
        ):
            gap, action = GapType.FALSE_PREMISE.value, Action.CHALLENGE.value
            reasons.append("false premise or contradiction")
        elif (f.get("risk_high", 0) or f.get("q_has_high_risk_keywords", 0)) and f.get("ev_sufficiency_score", 0) < 0.18:
            gap, action = GapType.HIGH_RISK.value, Action.ABSTAIN.value
            reasons.append("high-risk domain")
        elif f.get("q_has_time_words", 0) and (
            base.action_pred[i] == Action.RETRIEVE.value or f.get("ev_sufficiency_score", 0) < 0.24
        ):
            gap, action = GapType.TIME_SENSITIVE.value, Action.RETRIEVE.value
            reasons.append("time-sensitive query")
        elif (
            use_evidence_verifier
            and include_evidence
            and f.get("ev_sufficiency_score", 0) < 0.05
            and base.action_pred[i] == Action.ANSWER.value
        ):
            if f.get("ev_contradiction_proxy", 0) > 0.5 and f.get("q_asks_support_judgment", 0):
                gap, action = base.gap_pred[i], Action.ABSTAIN.value
                reasons.append("given evidence does not support the affirmative candidate")
            elif f.get("q_has_subjective_words", 0) or f.get("q_has_condition_words", 0):
                gap, action = GapType.USER_INFO_MISSING.value, Action.ASK.value
                reasons.append("missing user constraints")
            else:
                gap, action = GapType.EVIDENCE_MISSING.value, Action.RETRIEVE.value
                reasons.append("insufficient evidence score")
        elif use_gap_classifier and gap:
            action = base.action_pred[i]
            reasons.append(f"gap classifier predicted {gap}; action classifier predicted {action}")
        else:
            reasons.append("direct action classifier")
        if action == Action.ASK.value and not use_question_ranker:
            reasons.append("ask without utility ranker")
        gaps.append(gap or base.gap_pred[i])
        actions.append(action)
        explanations.append("; ".join(reasons))
    logging.getLogger(__name__).info("%s predictions generated", method_name)
    return PredictionResult(method_name, gaps, actions, explanations=explanations)


def prompted_llm_baseline(records: list[dict[str, Any]], client: Any, api_available: bool) -> PredictionResult:
    if not api_available:
        return PredictionResult(
            "Prompted LLM Baseline",
            [GapType.SUFFICIENT.value] * len(records),
            [Action.ANSWER.value] * len(records),
            status="skipped_api_unavailable",
            explanations=["API unavailable; baseline skipped"] * len(records),
        )
    gaps: list[str] = []
    actions: list[str] = []
    explanations: list[str] = []
    failures = 0
    system = (
        "You are evaluating a QA system's decision policy. Return only strict JSON in the final message content. "
        "Do not include chain-of-thought, markdown, or prose outside the JSON object. "
        f"`gap_type` must be one of: {', '.join(GAP_TYPES)}. "
        f"`action` must be one of: {', '.join(ACTION_TYPES)}. "
        "Use `answer` only when the candidate answer is safe and sufficiently supported; "
        "use `ask` when user constraints or ambiguity are missing; "
        "use `retrieve` when evidence is missing or freshness is required; "
        "use `abstain` for high-risk expert judgment; "
        "use `challenge_premise` when the question premise is unsupported or contradicted."
    )
    for record in records:
        payload = {
            "query": record["user_initial_query"],
            "dialogue_context": record["dialogue_context"],
            "retrieved_evidence": record["retrieved_evidence"],
            "candidate_answer": record["candidate_answer"],
        }
        result = client.chat_json(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        "Classify this sample and return JSON with keys "
                        "`gap_type`, `action`, and `explanation`.\n"
                        + json_dumps(payload)
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=2048,
            cache_key=f"prompted_llm_{record['id']}",
        )
        gap = _normalize_gap(result.get("gap_type") if isinstance(result, dict) else None)
        action = _normalize_action(result.get("action") if isinstance(result, dict) else None)
        explanation = result.get("explanation") if isinstance(result, dict) else None
        if gap not in GAP_TYPES:
            gap = GapType.SUFFICIENT.value
            failures += 1
        if action not in ACTION_TYPES:
            action = action_from_gap(gap)
            failures += 1
        gaps.append(gap)
        actions.append(action)
        explanations.append(str(explanation or "DeepSeek prediction"))
    status = "ok" if failures == 0 else f"partial_api_or_parse_failures_{failures}"
    return PredictionResult("Prompted LLM Baseline", gaps, actions, status=status, explanations=explanations)


def _normalize_gap(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "insufficient_information": GapType.EVIDENCE_MISSING.value,
        "missing_evidence": GapType.EVIDENCE_MISSING.value,
        "needs_retrieval": GapType.EVIDENCE_MISSING.value,
        "needs_clarification": GapType.USER_INFO_MISSING.value,
        "user_information_missing": GapType.USER_INFO_MISSING.value,
        "ambiguous": GapType.AMBIGUOUS.value,
        "false_assumption": GapType.FALSE_PREMISE.value,
        "outdated_information": GapType.TIME_SENSITIVE.value,
        "expert_needed": GapType.HIGH_RISK.value,
    }
    return normalized if normalized in GAP_TYPES else aliases.get(normalized)


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
        "challenge": Action.CHALLENGE.value,
        "correct_premise": Action.CHALLENGE.value,
    }
    return normalized if normalized in ACTION_TYPES else aliases.get(normalized)


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)
