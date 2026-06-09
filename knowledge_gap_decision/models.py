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
        if f["flag_false_premise"] or f["ev_contradiction_proxy"] > 0.5:
            gap, action, reason = GapType.FALSE_PREMISE.value, Action.CHALLENGE.value, "contradictory evidence or false-premise cue"
        elif f["risk_high"]:
            gap, action, reason = GapType.HIGH_RISK.value, Action.ABSTAIN.value, "high-risk keyword"
        elif f["q_has_time_words"] or f["flag_time_sensitive"]:
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
        if f["q_has_false_premise_pattern"] or f["flag_false_premise"]:
            gap, action, reason = GapType.FALSE_PREMISE.value, Action.CHALLENGE.value, "false-premise pattern"
        elif f["risk_high"]:
            gap, action, reason = GapType.HIGH_RISK.value, Action.ABSTAIN.value, "high-risk domain"
        elif f["q_has_time_words"] or f["flag_time_sensitive"]:
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
        if f.get("flag_false_premise", 0) or (
            include_evidence and f.get("ev_contradiction_proxy", 0) > 0.5 and f.get("q_has_false_premise_pattern", 0)
        ):
            gap, action = GapType.FALSE_PREMISE.value, Action.CHALLENGE.value
            reasons.append("false premise or contradiction")
        elif f.get("risk_high", 0):
            gap, action = GapType.HIGH_RISK.value, Action.ABSTAIN.value
            reasons.append("high-risk domain")
        elif f.get("flag_time_sensitive", 0) or f.get("q_has_time_words", 0):
            gap, action = GapType.TIME_SENSITIVE.value, Action.RETRIEVE.value
            reasons.append("time-sensitive query")
        elif use_evidence_verifier and include_evidence and f.get("ev_sufficiency_score", 0) < 0.05:
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


def prompted_llm_placeholder(records: list[dict[str, Any]], api_available: bool) -> PredictionResult:
    if not api_available:
        return PredictionResult(
            "Prompted LLM Baseline",
            [GapType.SUFFICIENT.value] * len(records),
            [Action.ANSWER.value] * len(records),
            status="skipped_api_unavailable",
            explanations=["API unavailable; baseline skipped"] * len(records),
        )
    # The project keeps the interface but avoids costly batch calls in default runs.
    return PredictionResult(
        "Prompted LLM Baseline",
        [GapType.SUFFICIENT.value] * len(records),
        [Action.ANSWER.value] * len(records),
        status="implemented_not_run_default",
        explanations=["GLM baseline implemented; not run in default offline experiment"] * len(records),
    )
