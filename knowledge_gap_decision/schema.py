from dataclasses import dataclass
from enum import Enum
from typing import Any


class GapType(str, Enum):
    SUFFICIENT = "sufficient_information"
    USER_INFO_MISSING = "user_info_missing"
    EVIDENCE_MISSING = "evidence_missing"
    AMBIGUOUS = "ambiguous_question"
    FALSE_PREMISE = "false_premise"
    TIME_SENSITIVE = "time_sensitive"
    HIGH_RISK = "high_risk_or_expert_needed"


class Action(str, Enum):
    ANSWER = "answer"
    ASK = "ask"
    RETRIEVE = "retrieve"
    ABSTAIN = "abstain"
    CHALLENGE = "challenge_premise"


GAP_TYPES = [x.value for x in GapType]
ACTION_TYPES = [x.value for x in Action]
EVIDENCE_LABELS = {"sufficient", "insufficient", "contradictory"}
RISK_LEVELS = {"low", "medium", "high"}

REQUIRED_FIELDS = {
    "id",
    "source",
    "group_id",
    "user_initial_query",
    "dialogue_context",
    "retrieved_evidence",
    "candidate_answer",
    "gap_type",
    "gold_action",
    "gold_clarifying_question",
    "final_answer",
    "required_slots",
    "evidence_sufficiency_label",
    "false_premise_flag",
    "time_sensitive_flag",
    "risk_level",
    "metadata",
}


@dataclass(frozen=True)
class ValidationErrorInfo:
    sample_id: str
    field: str
    message: str


def validate_record(record: dict[str, Any]) -> list[ValidationErrorInfo]:
    sample_id = str(record.get("id", "<missing>"))
    errors: list[ValidationErrorInfo] = []
    missing = REQUIRED_FIELDS - set(record)
    for field in sorted(missing):
        errors.append(ValidationErrorInfo(sample_id, field, "missing required field"))
    if errors:
        return errors

    string_fields = [
        "id",
        "source",
        "group_id",
        "user_initial_query",
        "dialogue_context",
        "candidate_answer",
        "gap_type",
        "gold_action",
        "gold_clarifying_question",
        "final_answer",
        "evidence_sufficiency_label",
        "risk_level",
    ]
    for field in string_fields:
        if not isinstance(record[field], str):
            errors.append(ValidationErrorInfo(sample_id, field, "must be a string"))

    if record.get("gap_type") not in GAP_TYPES:
        errors.append(ValidationErrorInfo(sample_id, "gap_type", "unknown gap type"))
    if record.get("gold_action") not in ACTION_TYPES:
        errors.append(ValidationErrorInfo(sample_id, "gold_action", "unknown action"))
    if record.get("evidence_sufficiency_label") not in EVIDENCE_LABELS:
        errors.append(ValidationErrorInfo(sample_id, "evidence_sufficiency_label", "unknown label"))
    if record.get("risk_level") not in RISK_LEVELS:
        errors.append(ValidationErrorInfo(sample_id, "risk_level", "unknown risk level"))
    if not isinstance(record.get("retrieved_evidence"), list) or not all(
        isinstance(x, str) for x in record.get("retrieved_evidence", [])
    ):
        errors.append(ValidationErrorInfo(sample_id, "retrieved_evidence", "must be list[str]"))
    if not isinstance(record.get("required_slots"), list) or not all(
        isinstance(x, str) for x in record.get("required_slots", [])
    ):
        errors.append(ValidationErrorInfo(sample_id, "required_slots", "must be list[str]"))
    if not isinstance(record.get("false_premise_flag"), bool):
        errors.append(ValidationErrorInfo(sample_id, "false_premise_flag", "must be bool"))
    if not isinstance(record.get("time_sensitive_flag"), bool):
        errors.append(ValidationErrorInfo(sample_id, "time_sensitive_flag", "must be bool"))
    if not isinstance(record.get("metadata"), dict):
        errors.append(ValidationErrorInfo(sample_id, "metadata", "must be dict"))

    return errors


def validate_dataset(records: list[dict[str, Any]]) -> None:
    errors: list[ValidationErrorInfo] = []
    seen_ids: set[str] = set()
    for record in records:
        errors.extend(validate_record(record))
        sample_id = str(record.get("id"))
        if sample_id in seen_ids:
            errors.append(ValidationErrorInfo(sample_id, "id", "duplicate id"))
        seen_ids.add(sample_id)
    if errors:
        preview = "; ".join(f"{e.sample_id}.{e.field}: {e.message}" for e in errors[:10])
        raise ValueError(f"dataset validation failed: {preview}")


def action_from_gap(gap_type: str) -> str:
    mapping = {
        GapType.SUFFICIENT.value: Action.ANSWER.value,
        GapType.USER_INFO_MISSING.value: Action.ASK.value,
        GapType.EVIDENCE_MISSING.value: Action.RETRIEVE.value,
        GapType.AMBIGUOUS.value: Action.ASK.value,
        GapType.FALSE_PREMISE.value: Action.CHALLENGE.value,
        GapType.TIME_SENSITIVE.value: Action.RETRIEVE.value,
        GapType.HIGH_RISK.value: Action.ABSTAIN.value,
    }
    return mapping.get(gap_type, Action.ABSTAIN.value)
