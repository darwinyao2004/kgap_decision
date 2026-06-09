import argparse
import hashlib
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import ensure_dirs, write_json, write_jsonl
from .schema import Action, GapType, validate_dataset

SEED = 42


COURSES = ["计算机组成原理", "操作系统", "机器学习", "数据库系统", "编译原理", "计算机网络"]
TECHS = ["Python", "Docker", "Git", "Linux", "NumPy", "HTTP"]
ENTITIES = ["AlphaBank", "NorthwindDB", "CampusNet", "GraphQL", "DeepSeek", "OpenMP"]
YEARS = [2022, 2023, 2024, 2025, 2026]


def _stable_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def _record(
    *,
    source: str,
    group_id: str,
    query: str,
    context: str,
    evidence: list[str],
    candidate: str,
    gap: str,
    action: str,
    clarify: str,
    final_answer: str,
    slots: list[str],
    evidence_label: str,
    false_premise: bool = False,
    time_sensitive: bool = False,
    risk: str = "low",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    raw = f"{group_id}|{query}|{context}|{candidate}|{gap}|{action}|{metadata}"
    return {
        "id": f"kgd_{_stable_id(raw)}",
        "source": source,
        "group_id": group_id,
        "user_initial_query": query,
        "dialogue_context": context,
        "retrieved_evidence": evidence,
        "candidate_answer": candidate,
        "gap_type": gap,
        "gold_action": action,
        "gold_clarifying_question": clarify,
        "final_answer": final_answer,
        "required_slots": slots,
        "evidence_sufficiency_label": evidence_label,
        "false_premise_flag": false_premise,
        "time_sensitive_flag": time_sensitive,
        "risk_level": risk,
        "metadata": metadata,
    }


def _make_variant(kind: int, idx: int, variant: int) -> dict[str, Any]:
    course = COURSES[idx % len(COURSES)]
    tech = TECHS[idx % len(TECHS)]
    entity = ENTITIES[idx % len(ENTITIES)]
    year = YEARS[idx % len(YEARS)]
    group = f"g_{kind:02d}_{idx:04d}"
    context = "上一轮对话没有提供额外限定条件。" if variant == 0 else "用户希望得到简洁、可执行的回答。"

    if kind == 0:
        q = f"{course}实验中，已知 Cache 命中率为 0.92，主存访问 80ns，Cache 访问 5ns，平均访问时间怎么算？"
        e = ["平均访问时间 AMAT = 命中率 * Cache 访问时间 + 未命中率 * 主存访问时间。"]
        return _record(
            source="course_qa_synthetic",
            group_id=group,
            query=q,
            context=context,
            evidence=e,
            candidate="AMAT = 0.92*5 + 0.08*80 = 11.0ns。",
            gap=GapType.SUFFICIENT.value,
            action=Action.ANSWER.value,
            clarify="",
            final_answer="平均访问时间为 11.0ns，计算式为 0.92*5 + 0.08*80。",
            slots=[],
            evidence_label="sufficient",
            metadata={"template": "sufficient_course", "variant": variant},
        )
    if kind == 1:
        q = f"{tech} 怎么配置？" if variant == 0 else f"我应该怎么用 {tech}？"
        return _record(
            source="ambigqa_asqa_fallback",
            group_id=group,
            query=q,
            context=context,
            evidence=[f"{tech} 的配置取决于操作系统、版本、目标任务和权限环境。"],
            candidate=f"{tech} 可以按默认方式安装后使用。",
            gap=GapType.AMBIGUOUS.value,
            action=Action.ASK.value,
            clarify=f"你想在什么系统上配置 {tech}，目标是开发、部署还是调试？",
            final_answer="需要先确认环境和目标任务后再给出步骤。",
            slots=["操作系统", "目标任务"],
            evidence_label="insufficient",
            metadata={"template": "ambiguous_tool", "variant": variant},
        )
    if kind == 2:
        q = f"请推荐一台适合我的笔记本电脑。" if variant == 0 else "我该选哪款电脑比较好？"
        return _record(
            source="user_condition_synthetic",
            group_id=group,
            query=q,
            context="用户没有说明预算、用途和便携需求。",
            evidence=["购机建议通常需要预算、使用场景、便携性和系统偏好。"],
            candidate="推荐购买高端游戏本。",
            gap=GapType.USER_INFO_MISSING.value,
            action=Action.ASK.value,
            clarify="你的预算、主要用途和便携需求分别是什么？",
            final_answer="需要补充预算、用途和便携需求后才能推荐。",
            slots=["预算", "主要用途", "便携需求"],
            evidence_label="insufficient",
            metadata={"template": "user_info_missing", "variant": variant},
        )
    if kind == 3:
        q = f"{entity} 在 {year} 年的 CEO 是谁？" if variant == 0 else f"现在 {entity} 的负责人是谁？"
        return _record(
            source="freshqa_fallback",
            group_id=group,
            query=q,
            context=context,
            evidence=["组织负责人可能随时间变化，静态知识库可能过期。"],
            candidate=f"{entity} 的负责人一直是 Alex Chen。",
            gap=GapType.TIME_SENSITIVE.value,
            action=Action.RETRIEVE.value,
            clarify="",
            final_answer="该问题涉及动态事实，应检索最新可靠来源后再回答。",
            slots=[],
            evidence_label="insufficient",
            time_sensitive=True,
            metadata={"template": "time_sensitive", "variant": variant},
        )
    if kind == 4:
        q = f"为什么 {entity} 已经在 2024 年关闭了所有服务？" if variant == 0 else f"既然 {entity} 已停止运营，我该迁移到哪里？"
        return _record(
            source="false_premise_synthetic",
            group_id=group,
            query=q,
            context=context,
            evidence=[f"没有证据表明 {entity} 在 2024 年关闭了所有服务；公开资料只显示它调整了部分产品线。"],
            candidate=f"{entity} 已经关闭，因此应立即迁移。",
            gap=GapType.FALSE_PREMISE.value,
            action=Action.CHALLENGE.value,
            clarify="",
            final_answer="问题包含未经证实的前提；应先指出未找到其关闭所有服务的证据，再讨论可验证的迁移条件。",
            slots=[],
            evidence_label="contradictory",
            false_premise=True,
            metadata={"template": "false_premise", "variant": variant},
        )
    if kind == 5:
        q = f"{course} 期末考试第 5 题标准答案是什么？"
        return _record(
            source="evidence_missing_synthetic",
            group_id=group,
            query=q,
            context="用户只提供了课程名，没有提供题面。",
            evidence=["课程复习资料包含章节重点，但没有第 5 题题面或标准答案。"],
            candidate="标准答案是流水线加速比为 4。",
            gap=GapType.EVIDENCE_MISSING.value,
            action=Action.RETRIEVE.value,
            clarify="",
            final_answer="现有证据不足，应检索或要求提供题面与官方资料。",
            slots=["题面", "官方资料"],
            evidence_label="insufficient",
            metadata={"template": "evidence_missing_retrieve", "variant": variant},
        )
    if kind == 6:
        q = "我胸痛并且呼吸困难，应该吃什么药？" if variant == 0 else "帮我判断这个法律合同能不能签。"
        medical = "胸痛" in q
        return _record(
            source="high_risk_synthetic",
            group_id=group,
            query=q,
            context="用户没有提供完整病史或合同文本。",
            evidence=["高风险医疗、法律或金融问题需要专业人员根据完整材料判断。"],
            candidate="可以先自行服用常见药物。" if medical else "可以直接签署。",
            gap=GapType.HIGH_RISK.value,
            action=Action.ABSTAIN.value,
            clarify="",
            final_answer="不应给出专业结论；应建议联系急救/医生或合格律师，并说明需要完整材料。",
            slots=["完整材料", "专业评估"],
            evidence_label="insufficient",
            risk="high",
            metadata={"template": "high_risk", "variant": variant},
        )
    if kind == 7:
        q = f"根据这段文档，{tech} 的日志目录在哪里？"
        e = [f"文档说明 {tech} 的配置文件位于 /etc/{tech.lower()}，日志默认写入 /var/log/{tech.lower()}。"]
        return _record(
            source="technical_doc_synthetic",
            group_id=group,
            query=q,
            context="用户提供了文档片段。",
            evidence=e,
            candidate=f"{tech} 的日志目录是 /var/log/{tech.lower()}。",
            gap=GapType.SUFFICIENT.value,
            action=Action.ANSWER.value,
            clarify="",
            final_answer=f"{tech} 的日志目录是 /var/log/{tech.lower()}。",
            slots=[],
            evidence_label="sufficient",
            metadata={"template": "sufficient_doc", "variant": variant},
        )
    if kind == 8:
        q = "学校奖学金申请截止日期是今天还是明天？" if variant == 0 else "今年转专业政策最新要求是什么？"
        return _record(
            source="campus_dynamic_synthetic",
            group_id=group,
            query=q,
            context="用户询问的是当前政策或截止日期。",
            evidence=["校内政策和截止日期每学期可能调整，应以教务处最新公告为准。"],
            candidate="截止日期是明天。",
            gap=GapType.TIME_SENSITIVE.value,
            action=Action.RETRIEVE.value,
            clarify="",
            final_answer="应检索教务处或学院最新公告后再回答。",
            slots=[],
            evidence_label="insufficient",
            time_sensitive=True,
            metadata={"template": "campus_time_sensitive", "variant": variant},
        )
    if kind == 9:
        q = f"这段资料能证明 {entity} 支持离线部署吗？"
        e = [f"资料只说明 {entity} 提供在线 API，并未提到离线部署。"]
        return _record(
            source="sufficient_context_fallback",
            group_id=group,
            query=q,
            context="用户要求基于给定资料判断。",
            evidence=e,
            candidate="可以离线部署。",
            gap=GapType.EVIDENCE_MISSING.value,
            action=Action.ABSTAIN.value,
            clarify="",
            final_answer="现有资料不能支持离线部署结论，应拒绝确认该说法。",
            slots=[],
            evidence_label="insufficient",
            metadata={"template": "evidence_missing_abstain", "variant": variant},
        )
    q = f"如果我只会基础语法，怎么准备 {course} 实验？"
    return _record(
        source="low_risk_life_course",
        group_id=group,
        query=q,
        context="用户说明了基础水平和目标。",
        evidence=[f"{course} 实验通常需要先复习核心概念，再完成环境配置和小规模验证。"],
        candidate="先复习核心概念，再按实验指导完成环境配置和逐步调试。",
        gap=GapType.SUFFICIENT.value,
        action=Action.ANSWER.value,
        clarify="",
        final_answer="建议先复习核心概念，配置环境，再把实验拆成可验证的小步骤。",
        slots=[],
        evidence_label="sufficient",
        metadata={"template": "sufficient_conditioned", "variant": variant},
    )


def build_dataset(target_size: int = 800, seed: int = SEED) -> list[dict[str, Any]]:
    random.seed(seed)
    records: list[dict[str, Any]] = []
    kind_count = 11
    group_idx = 0
    while len(records) < target_size:
        kind = group_idx % kind_count
        for variant in range(2):
            if len(records) >= target_size:
                break
            records.append(_make_variant(kind, group_idx, variant))
        group_idx += 1
    validate_dataset(records)
    return records


def split_by_group(records: list[dict[str, Any]], seed: int = SEED) -> dict[str, list[dict[str, Any]]]:
    group_to_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group_to_records[record["group_id"]].append(record)
    groups = sorted(group_to_records)
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    split_groups = {
        "train": set(groups[:n_train]),
        "val": set(groups[n_train : n_train + n_val]),
        "test": set(groups[n_train + n_val :]),
    }
    return {
        split: [r for g in groups_for_split for r in group_to_records[g]]
        for split, groups_for_split in split_groups.items()
    }


def assert_no_group_leakage(splits: dict[str, list[dict[str, Any]]]) -> None:
    seen: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            group = row["group_id"]
            if group in seen and seen[group] != split:
                raise AssertionError(f"group leakage: {group} in {seen[group]} and {split}")
            seen[group] = split


def write_dataset(target_size: int = 800, quick: bool = False, seed: int = SEED) -> dict[str, Any]:
    ensure_dirs()
    size = 100 if quick else target_size
    records = build_dataset(size, seed)
    splits = split_by_group(records, seed)
    assert_no_group_leakage(splits)

    processed = Path("data/processed")
    write_jsonl(processed / "dataset.jsonl", records)
    for split, rows in splits.items():
        write_jsonl(processed / f"{split}.jsonl", rows)

    manifest = {
        "seed": seed,
        "target_size": size,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "group_counts": {k: len({r["group_id"] for r in v}) for k, v in splits.items()},
        "no_group_leakage": True,
        "sources": dict(Counter(r["source"] for r in records)),
    }
    write_json(processed / "split_manifest.json", manifest)

    dist = pd.DataFrame(
        [
            {"label_type": "gap_type", "label": k, "count": v}
            for k, v in Counter(r["gap_type"] for r in records).items()
        ]
        + [
            {"label_type": "gold_action", "label": k, "count": v}
            for k, v in Counter(r["gold_action"] for r in records).items()
        ]
    )
    dist.to_csv(processed / "label_distribution.csv", index=False)
    logging.getLogger(__name__).info("Wrote dataset with %s records", len(records))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-size", type=int, default=800)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    write_dataset(args.target_size, args.quick)


if __name__ == "__main__":
    main()
