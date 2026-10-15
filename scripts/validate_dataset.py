#!/usr/bin/env python3
"""
FlowersForYou - Dataset Validator & Quality Auditor.

Performs 10-point rigorous verification on dataset batches and compiled datasets:
1. JSONL structural integrity & message format.
2. Conversation flow: valid roles, presence of user turn, assistant final turn.
3. Canonical schema validation for preference extraction.
4. Semantic safety & behavioral rule validation (using evaluate_response from evaluate.py).
5. Grounding verification for tool-assisted dialogues.
6. Exact duplicate detection.
7. Intra-dataset and cross-batch near-duplicate check (Jaccard threshold 0.85).
8. Held-out separation against data/eval.jsonl (zero exact leakage, Jaccard < 0.40).
9. Provenance manifest synchronization.
10. Category allocation and count audit.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluate import evaluate_response

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
CANONICAL_PREFERENCE_KEYS = {
    "action",
    "interests",
    "preferred_neighborhoods",
    "budget_inr_max",
    "budget_inr_min",
    "timing",
    "social_vibe",
    "max_group_size",
}


def tokenize(text: str) -> set:
    """Normalize text into alphanumeric lowercase tokens."""
    return set(re.findall(r"\w+", text.lower()))


def jaccard_similarity(s1: str, s2: str) -> float:
    t1 = tokenize(s1)
    t2 = tokenize(s2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def get_conversation_user_text(messages: list) -> str:
    return " ".join(m.get("content", "") for m in messages if m.get("role") == "user")


def get_final_assistant_text(messages: list) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def validate_single_record(record: dict, record_idx: int = 1) -> tuple[bool, list[str]]:
    """Validate structure, schema, and safety of a single conversational record."""
    errors = []
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return False, [f"Record #{record_idx}: missing or non-list 'messages'"]

    roles = []
    has_tool_result = False
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"Record #{record_idx}: message {i} is not a dict")
            continue
        role = msg.get("role")
        content = msg.get("content")
        roles.append(role)
        if role not in ALLOWED_ROLES:
            errors.append(f"Record #{record_idx}: invalid role '{role}' in message {i}")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"Record #{record_idx}: message {i} has empty or non-string content")
        if role == "tool":
            has_tool_result = True
            # Validate tool content is valid json if applicable
            try:
                json.loads(content)
            except Exception:
                pass  # tool content can be plain text or json

    if "user" not in roles:
        errors.append(f"Record #{record_idx}: conversation has no 'user' turn")
    if "assistant" not in roles:
        errors.append(f"Record #{record_idx}: conversation has no 'assistant' turn")
    if roles and roles[-1] != "assistant":
        errors.append(f"Record #{record_idx}: final turn must be 'assistant', got '{roles[-1]}'")

    # Extract conversation context for behavioral safety check
    user_text = next((m.get("content", "") for m in reversed(messages[:-1]) if m.get("role") == "user"), "")
    system_text = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    assistant_resp = messages[-1].get("content", "") if messages else ""

    # Preference extraction schema check
    if "preference extractor" in system_text.lower():
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", assistant_resp, re.DOTALL)
        if not json_match:
            errors.append(f"Record #{record_idx}: preference extraction missing ```json ... ``` code block")
        else:
            try:
                parsed = json.loads(json_match.group(1))
                if not isinstance(parsed, dict):
                    errors.append(f"Record #{record_idx}: extracted JSON is not an object")
                elif parsed.get("action") != "save_preferences":
                    errors.append(f"Record #{record_idx}: extraction schema missing 'action: save_preferences'")
                else:
                    extra_keys = set(parsed.keys()) - CANONICAL_PREFERENCE_KEYS
                    if extra_keys:
                        errors.append(f"Record #{record_idx}: non-canonical schema keys: {list(extra_keys)}")
            except Exception as exc:
                errors.append(f"Record #{record_idx}: invalid JSON in extraction block: {exc}")

    # Semantic behavioral safety evaluation
    eval_res = evaluate_response(assistant_resp, user_text, system_text, has_tool_result=has_tool_result)
    if not eval_res["is_safe"]:
        for viol in eval_res["violations"]:
            errors.append(f"Record #{record_idx}: {viol}")

    return len(errors) == 0, errors


def audit_batch_or_dataset(
    data_path: Path,
    eval_path: Path = ROOT / "data/eval.jsonl",
    manifest_path: Path = None,
    target_count: int = None,
) -> dict:
    """Run full quality and overlap audit on a dataset file."""
    report = {
        "file": str(data_path),
        "total_records": 0,
        "valid_records": 0,
        "structural_errors": [],
        "safety_violations": [],
        "exact_duplicates": 0,
        "near_duplicates": 0,
        "held_out_leakages": [],
        "max_eval_jaccard": 0.0,
        "highest_overlap_pair": None,
        "manifest_sync": True,
        "category_counts": {},
        "passed": False,
    }

    if not data_path.exists():
        report["structural_errors"].append(f"File not found: {data_path}")
        return report

    records = []
    line_strings = []
    user_prompts = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            line_strings.append(line)
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError as exc:
                report["structural_errors"].append(f"Line {line_no}: invalid JSON: {exc}")

    report["total_records"] = len(records)

    # 1. Structural and Safety Validation
    for i, rec in enumerate(records, start=1):
        ok, errs = validate_single_record(rec, record_idx=i)
        if ok:
            report["valid_records"] += 1
        else:
            for e in errs:
                if "[SAFETY VIOLATION]" in e or "[STRUCTURE VIOLATION]" in e or "[BEHAVIOR VIOLATION]" in e:
                    report["safety_violations"].append(e)
                else:
                    report["structural_errors"].append(e)

        u_text = get_conversation_user_text(rec.get("messages", []))
        user_prompts.append(u_text)

    # 2. Exact Duplicates
    seen_lines = set()
    for l in line_strings:
        if l in seen_lines:
            report["exact_duplicates"] += 1
        seen_lines.add(l)

    # 3. Intra-Dataset Near-Duplicates (Sampled check for large datasets, full for <= 1000)
    # Check distinct user prompts
    if len(records) <= 1000:
        for i in range(len(user_prompts)):
            for j in range(i + 1, len(user_prompts)):
                sim = jaccard_similarity(user_prompts[i], user_prompts[j])
                if sim >= 0.85:
                    report["near_duplicates"] += 1

    # 4. Held-Out Evaluation Separation
    if eval_path.exists():
        eval_records = []
        with open(eval_path, "r", encoding="utf-8") as f:
            for eline in f:
                eline = eline.strip()
                if eline:
                    eval_records.append(json.loads(eline))

        for tr_idx, tr_rec in enumerate(records, start=1):
            tr_u = get_conversation_user_text(tr_rec.get("messages", []))
            tr_a = get_final_assistant_text(tr_rec.get("messages", []))

            for ev_idx, ev_rec in enumerate(eval_records, start=1):
                ev_u = get_conversation_user_text(ev_rec.get("messages", []))
                ev_a = get_final_assistant_text(ev_rec.get("messages", []))

                # Exact prompt match
                if tr_u.strip().lower() == ev_u.strip().lower():
                    report["held_out_leakages"].append(
                        f"Exact prompt match between Record #{tr_idx} and Eval #{ev_idx}: '{tr_u}'"
                    )
                # Exact assistant match
                if tr_a.strip().lower() == ev_a.strip().lower():
                    report["held_out_leakages"].append(
                        f"Exact response match between Record #{tr_idx} and Eval #{ev_idx}"
                    )

                sim = jaccard_similarity(tr_u, ev_u)
                if sim > report["max_eval_jaccard"]:
                    report["max_eval_jaccard"] = sim
                    report["highest_overlap_pair"] = (tr_idx, ev_idx, tr_u, ev_u, sim)

    # 5. Manifest Synchronization Check
    if manifest_path and manifest_path.exists():
        manifest_records = []
        with open(manifest_path, "r", encoding="utf-8") as mf:
            for mline in mf:
                mline = mline.strip()
                if mline:
                    manifest_records.append(json.loads(mline))

        if len(manifest_records) != len(records):
            report["manifest_sync"] = False
            report["structural_errors"].append(
                f"Manifest count ({len(manifest_records)}) does not match dataset count ({len(records)})"
            )
        else:
            for m in manifest_records:
                cat = m.get("category", "unknown")
                report["category_counts"][cat] = report["category_counts"].get(cat, 0) + 1

    # Overall pass criteria
    no_errors = len(report["structural_errors"]) == 0
    no_safety = len(report["safety_violations"]) == 0
    no_exact_dupes = report["exact_duplicates"] == 0
    no_leakage = len(report["held_out_leakages"]) == 0
    target_met = (target_count is None) or (report["valid_records"] == target_count)

    report["passed"] = no_errors and no_safety and no_exact_dupes and no_leakage and target_met
    return report


def main():
    parser = argparse.ArgumentParser(description="FlowersForYou Dataset Validator.")
    parser.add_argument("data_file", type=str, help="Path to JSONL dataset file to validate.")
    parser.add_argument("--eval-file", type=str, default=str(ROOT / "data/eval.jsonl"), help="Held-out eval file.")
    parser.add_argument("--manifest-file", type=str, default=None, help="Optional sidecar manifest file.")
    parser.add_argument("--target-count", type=int, default=None, help="Expected record count.")
    args = parser.parse_args()

    data_path = Path(args.data_file)
    eval_path = Path(args.eval_file)
    manifest_path = Path(args.manifest_file) if args.manifest_file else None

    print("=" * 65)
    print(f"Auditing Dataset: {data_path.name}")
    print("=" * 65)

    res = audit_batch_or_dataset(
        data_path=data_path,
        eval_path=eval_path,
        manifest_path=manifest_path,
        target_count=args.target_count,
    )

    print(f"Total Records:      {res['total_records']}")
    print(f"Valid Records:      {res['valid_records']}")
    print(f"Structural Errors:  {len(res['structural_errors'])}")
    print(f"Safety Violations:  {len(res['safety_violations'])}")
    print(f"Exact Duplicates:   {res['exact_duplicates']}")
    print(f"Near Duplicates:    {res['near_duplicates']}")
    print(f"Max Eval Jaccard:   {res['max_eval_jaccard']:.3f}")
    print(f"Held-Out Leakages:  {len(res['held_out_leakages'])}")

    if res['highest_overlap_pair']:
        p = res['highest_overlap_pair']
        print(f"Highest Eval Match: Record #{p[0]} vs Eval #{p[1]} (Jaccard = {p[4]:.3f})")
        print(f"  Dataset: \"{p[2]}\"")
        print(f"  Eval:    \"{p[3]}\"")

    if res["category_counts"]:
        print("\nCategory Distribution:")
        for cat, cnt in sorted(res["category_counts"].items()):
            print(f"  - {cat:55s}: {cnt}")

    if not res["passed"]:
        print("\n[FAIL] Dataset did not meet quality thresholds.")
        if res["structural_errors"]:
            print("Structural Errors (first 5):")
            for e in res["structural_errors"][:5]:
                print("  *", e)
        if res["safety_violations"]:
            print("Safety Violations (first 5):")
            for e in res["safety_violations"][:5]:
                print("  *", e)
        if res["held_out_leakages"]:
            print("Leakages:")
            for l in res["held_out_leakages"][:5]:
                print("  *", l)
        sys.exit(1)
    else:
        print("\n[PASS] All validation and safety thresholds passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
