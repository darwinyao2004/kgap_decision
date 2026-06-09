import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .data_build import write_dataset
from .data_build import build_dataset, split_by_group
from .evaluate import (
    evaluate_prediction,
    per_class_rows,
    plot_confusion,
    plot_utility,
    significance_rows,
)
from .features import compute_features, feature_columns
from .io_utils import ensure_dirs, read_jsonl, setup_logging, write_json
from .models import (
    DualClassifier,
    TextClassifier,
    always_baseline,
    full_method_predict,
    prompted_llm_baseline,
    rag_threshold,
    self_consistency_baseline,
)
from .question_ranker import rank_questions
from .schema import ACTION_TYPES, GAP_TYPES
from .deepseek_client import DeepSeekClient


def _load_splits() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return read_jsonl("data/processed/train.jsonl"), read_jsonl("data/processed/val.jsonl"), read_jsonl("data/processed/test.jsonl")


def _error_suggestion(record: dict[str, Any], gold: str, pred_action: str) -> str:
    if gold == "abstain" and pred_action == "retrieve":
        return "现有证据已经不足以支持候选结论时，应优先拒绝确认，不必默认追加检索。"
    if gold == "retrieve" and pred_action == "answer":
        return "时间敏感或证据缺失样本需要提高检索优先级，避免静态答案提前输出。"
    if gold == "ask" and pred_action != "ask":
        slots = "、".join(record.get("required_slots") or ["关键条件"])
        return f"先追问 {slots}，再给出步骤或推荐。"
    if gold == "challenge_premise" and pred_action == "answer":
        return "对“为什么/既然/已经”等前提词增加核验分支，先说明前提未被证据支持。"
    if gold == "abstain" and pred_action == "answer":
        return "高风险或不可验证问题不应给出确定结论，需要保留拒答阈值。"
    return "检查该类样本的标签边界，并补充相邻动作的对比样本。"


def _write_error_analysis(records: list[dict[str, Any]], full_pred, features: pd.DataFrame) -> None:
    categories = {
        ("answer", "ask"): "answer 误判为 ask",
        ("ask", "answer"): "ask 误判为 answer",
        ("retrieve", "answer"): "retrieve 误判为 answer",
        ("challenge_premise", "answer"): "challenge_premise 误判为 answer",
        ("abstain", "answer"): "abstain 误判为 answer",
    }
    errors = []
    seen_signatures = set()
    for i, (record, pred_action, pred_gap) in enumerate(zip(records, full_pred.action_pred, full_pred.gap_pred)):
        gold = record["gold_action"]
        if gold == pred_action:
            continue
        f = features.iloc[i].to_dict()
        category = categories.get((gold, pred_action), "其他动作混淆")
        if record.get("false_premise_flag") and pred_action == "answer":
            category = "false premise compliance"
        if record.get("time_sensitive_flag") and pred_action != "retrieve":
            category = "time-sensitive 未触发 retrieve"
        signature = (record["user_initial_query"], gold, pred_action, category)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        errors.append(
            {
                "id": record["id"],
                "query": record["user_initial_query"],
                "gold_gap": record["gap_type"],
                "gold_action": gold,
                "pred_gap": pred_gap,
                "pred_action": pred_action,
                "key_features": {
                    "ev_sufficiency_score": round(float(f.get("ev_sufficiency_score", 0)), 3),
                    "ev_contradiction_proxy": round(float(f.get("ev_contradiction_proxy", 0)), 3),
                    "q_has_time_words": int(f.get("q_has_time_words", 0)),
                    "q_has_high_risk_keywords": int(f.get("q_has_high_risk_keywords", 0)),
                    "sc_self_consistency_score": round(float(f.get("sc_self_consistency_score", 0)), 3),
                },
                "error_category": category,
                "suggestion": _error_suggestion(record, gold, pred_action),
            }
        )
    errors = errors[:20]
    Path("results/error_analysis.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 错误分析", ""]
    if not errors:
        lines.append("Full Method 在当前测试集没有动作预测错误。这个结果不代表真实场景已经解决，主要说明当前合成模板的边界比较清楚。后续应加入真实问句或更接近真实噪声的改写样本继续检查。")
    else:
        counts = pd.Series([e["error_category"] for e in errors]).value_counts()
        lines.append("错误类别计数：")
        for label, count in counts.items():
            lines.append(f"- {label}: {count}")
        lines.append("")
    for idx, e in enumerate(errors, 1):
        lines.extend(
            [
                f"## 案例 {idx}: {e['error_category']}",
                f"- query: {e['query']}",
                f"- gold: {e['gold_gap']} / {e['gold_action']}",
                f"- predicted: {e['pred_gap']} / {e['pred_action']}",
                f"- key_features: {json.dumps(e['key_features'], ensure_ascii=False)}",
                f"- 改进建议: {e['suggestion']}",
                "",
            ]
        )
    Path("results/error_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def _write_seed_stability(target_size: int, quick: bool) -> None:
    size = 100 if quick else target_size
    rows = []
    for seed in [13, 42, 101]:
        records = build_dataset(size, seed)
        splits = split_by_group(records, seed)
        train_all = splits["train"] + splits["val"]
        test = splits["test"]
        train_features = compute_features(train_all, cache_samples=False)
        test_features = compute_features(test, cache_samples=False)
        pred = full_method_predict(train_features, train_all, test_features, method_name=f"Full Method seed {seed}")
        metric = evaluate_prediction("Full Method", test, pred.gap_pred, pred.action_pred, pred.status)
        rows.append(
            {
                "seed": seed,
                "test_size": len(test),
                "gap_type_macro_f1": metric["gap_type_macro_f1"],
                "action_macro_f1": metric["action_macro_f1"],
                "action_accuracy": metric["action_accuracy"],
                "wrong_answer_rate": metric["wrong_answer_rate"],
                "score": metric["score"],
            }
        )
    pd.DataFrame(rows).to_csv("results/repeated_seed_summary.csv", index=False)


def _select_full_config(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
) -> dict[str, Any]:
    train_features = compute_features(train_records, cache_samples=False)
    val_features = compute_features(val_records, cache_samples=False)
    candidates: list[tuple[str, dict[str, Any]]] = [
        ("all_modules", {}),
        ("no_evidence_sufficiency_verifier", {"use_evidence_verifier": False}),
        ("no_evidence_side_features", {"include_evidence": False}),
        ("no_gap_type_classifier", {"use_gap_classifier": False}),
    ]
    rows = []
    best_name = candidates[0][0]
    best_kwargs = candidates[0][1]
    best_metric: dict[str, Any] | None = None
    for name, kwargs in candidates:
        pred = full_method_predict(train_features, train_records, val_features, method_name=f"Full candidate {name}", **kwargs)
        metric = evaluate_prediction(name, val_records, pred.gap_pred, pred.action_pred, pred.status)
        rows.append({"candidate": name, **kwargs, **metric})
        if best_metric is None or (metric["action_macro_f1"], metric["score"]) > (
            best_metric["action_macro_f1"],
            best_metric["score"],
        ):
            best_name = name
            best_kwargs = kwargs
            best_metric = metric
    selection = {
        "selected_candidate": best_name,
        "selected_kwargs": best_kwargs,
        "validation_rows": rows,
    }
    selection = json.loads(json.dumps(selection, default=float))
    write_json("results/full_method_selection.json", selection)
    pd.DataFrame(rows).to_csv("results/full_method_selection.csv", index=False)
    return best_kwargs


def run(
    target_size: int = 800,
    quick: bool = False,
    probe_api: bool = True,
    *,
    use_llm_data: bool = True,
    refresh_llm_data: bool = False,
) -> dict[str, Any]:
    setup_logging()
    ensure_dirs()
    logger = logging.getLogger(__name__)
    manifest = write_dataset(target_size, quick, use_llm=use_llm_data, refresh_llm_cache=refresh_llm_data)
    train, val, test = _load_splits()
    train_all = train + val
    logger.info("Loaded splits: train=%s val=%s test=%s", len(train), len(val), len(test))

    llm_client = DeepSeekClient()
    api_available = False
    if probe_api:
        try:
            api_available = llm_client.is_available()
        except Exception as exc:
            logger.warning("API probe failed and will use offline fallback: %s", type(exc).__name__)
    api_status = {
        "provider": "deepseek",
        "model": llm_client.model,
        "deepseek_api_key_present": bool(llm_client.enabled),
        "deepseek_api_available": api_available,
        "api_key_present": bool(llm_client.enabled),
        "api_available": api_available,
        "note": "DeepSeek API did not pass strict JSON probe; offline fallback used." if not api_available else "DeepSeek API available.",
    }
    write_json("results/api_status.json", api_status)

    selected_full_kwargs = _select_full_config(train, val)
    train_features = compute_features(train_all)
    test_features = compute_features(test)
    train_features.to_csv("data/processed/features_train.csv", index=False)
    test_features.to_csv("data/processed/features_test.csv", index=False)

    results = []
    per_class = []
    preds_for_sig = {}
    pred_objects = []

    baselines = [
        always_baseline(test, "Always Answer", "answer", train_all),
        always_baseline(test, "Always Ask", "ask", train_all),
        rag_threshold(test, test_features),
        self_consistency_baseline(test, test_features),
        prompted_llm_baseline(test, llm_client, api_available),
    ]

    cols = feature_columns(train_features)
    for kind, name in [("logistic", "Logistic Regression"), ("rf", "Random Forest"), ("gbdt", "GBDT")]:
        model = DualClassifier(kind, cols).fit(train_features, train_all)
        baselines.append(model.predict(test_features, name))

    text_model = TextClassifier().fit(train_all)
    baselines.append(text_model.predict(test))
    full = full_method_predict(train_features, train_all, test_features, **selected_full_kwargs)
    baselines.append(full)

    for pred in baselines:
        metric = evaluate_prediction(pred.method, test, pred.gap_pred, pred.action_pred, pred.status)
        results.append(metric)
        per_class.extend(per_class_rows(pred.method, test, pred.gap_pred, pred.action_pred))
        pred_objects.append(pred)
        if pred.status == "ok":
            preds_for_sig[pred.method] = pred.action_pred

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv("results/metrics_summary.csv", index=False)
    pd.DataFrame(per_class).to_csv("results/per_class_metrics.csv", index=False)

    full_pred = full
    pred_rows = []
    explanations = full_pred.explanations or [""] * len(test)
    for r, gp, ap, ex in zip(test, full_pred.gap_pred, full_pred.action_pred, explanations):
        pred_rows.append(
            {
                "id": r["id"],
                "query": r["user_initial_query"],
                "gold_gap_type": r["gap_type"],
                "gold_action": r["gold_action"],
                "pred_gap_type": gp,
                "pred_action": ap,
                "explanation": ex,
            }
        )
    pd.DataFrame(pred_rows).to_csv("results/predictions_test.csv", index=False)

    plot_confusion([r["gold_action"] for r in test], full_pred.action_pred, ACTION_TYPES, "results/confusion_matrix_action.png", "Action confusion matrix")
    plot_confusion([r["gap_type"] for r in test], full_pred.gap_pred, GAP_TYPES, "results/confusion_matrix_gap_type.png", "Gap type confusion matrix")
    plot_utility(metrics_df, "results/utility_comparison.png")

    ablation_specs = [
        ("Full - no gap type classifier", {"use_gap_classifier": False}),
        ("Full - no evidence sufficiency verifier", {"use_evidence_verifier": False}),
        ("Full - no self-consistency features", {"include_uncertainty": False}),
        ("Full - no question utility ranker", {"use_question_ranker": False}),
        ("Full - no evidence-side features", {"include_evidence": False}),
        ("Full - no model uncertainty features", {"include_uncertainty": False}),
    ]
    ablation_rows = []
    full_metric = metrics_df[metrics_df["method"] == "Full Method"].iloc[0].to_dict()
    for name, kwargs in ablation_specs:
        pred = full_method_predict(train_features, train_all, test_features, method_name=name, **kwargs)
        metric = evaluate_prediction(name, test, pred.gap_pred, pred.action_pred, pred.status)
        ablation_rows.append(
            {
                "variant": name,
                "action_macro_f1": metric["action_macro_f1"],
                "gap_type_macro_f1": metric["gap_type_macro_f1"],
                "score": metric["score"],
                "delta_action_macro_f1": metric["action_macro_f1"] - full_metric["action_macro_f1"],
                "delta_score": metric["score"] - full_metric["score"],
            }
        )
    pd.DataFrame(ablation_rows).to_csv("results/ablation_summary.csv", index=False)

    sig = significance_rows(test, preds_for_sig) if "Full Method" in preds_for_sig else []
    pd.DataFrame(sig).to_csv("results/significance_tests.csv", index=False)
    rank_df = rank_questions(test, full_pred.action_pred)
    rank_df.to_csv("results/question_ranker_eval.csv", index=False)
    _write_error_analysis(test, full_pred, test_features)
    _write_seed_stability(target_size, quick)

    experiment_log = {
        "mode": "quick" if quick else "full",
        "manifest": manifest,
        "api_status": api_status,
        "best_method_by_score": metrics_df.sort_values("score", ascending=False).iloc[0]["method"],
    }
    write_json("reports/experiment_log.json", experiment_log)
    Path("reports/experiment_log.md").write_text(
        "\n".join(
            [
                "# 实验日志",
                f"- mode: {experiment_log['mode']}",
                f"- split_sizes: {manifest['split_sizes']}",
                f"- api_available: {api_available}",
                f"- best_method_by_score: {experiment_log['best_method_by_score']}",
            ]
        ),
        encoding="utf-8",
    )
    logger.info("Experiment finished")
    return experiment_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-size", type=int, default=800)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-api-probe", action="store_true")
    parser.add_argument("--offline-data", action="store_true")
    parser.add_argument("--refresh-llm-data", action="store_true")
    args = parser.parse_args()
    run(
        args.target_size,
        args.quick,
        probe_api=not args.no_api_probe,
        use_llm_data=not args.offline_data,
        refresh_llm_data=args.refresh_llm_data,
    )


if __name__ == "__main__":
    main()
