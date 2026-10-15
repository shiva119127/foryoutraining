#!/usr/bin/env python3
"""
Comprehensive Validation & Leakage Audit for Phase 7 Pan-India Benchmark.

Audits:
1. JSONL parsing and required message structures.
2. Canonical schema compliance for preference extraction examples.
3. Category balance, geographic distribution, and feature representation.
4. Exact duplicate and semantic near-duplicate checks within eval_pan_india.jsonl.
5. Cross-dataset leakage against train_expanded_v2.jsonl, train.jsonl, eval.jsonl, and eval_expanded.jsonl.
6. Evaluation harness compatibility check against evaluate.py's evaluate_response().
"""

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data"
PAN_INDIA_FILE = DATA_DIR / "eval_pan_india.jsonl"

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
    """Normalize text into lowercase alphanumeric tokens."""
    return set(re.findall(r"\w+", text.lower()))


def jaccard_similarity(s1: str, s2: str) -> float:
    t1 = tokenize(s1)
    t2 = tokenize(s2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def get_conversation_user_text(messages: list) -> str:
    """Concatenate all user turns in a conversation."""
    return " ".join(m.get("content", "") for m in messages if m.get("role") == "user")


def get_final_assistant_text(messages: list) -> str:
    """Get the target assistant completion."""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def run_benchmark_audit():
    print("=" * 70)
    print("FlowersForYou Phase 7 — Pan-India Evaluation Benchmark Forensic Audit")
    print("=" * 70)

    if not PAN_INDIA_FILE.exists():
        print(f"[FAIL] Benchmark file not found: {PAN_INDIA_FILE}")
        sys.exit(1)

    records = []
    errors = []
    warnings = []

    # 1. Parsing & Structural Integrity
    with open(PAN_INDIA_FILE, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append((line_no, rec))
            except Exception as exc:
                errors.append(f"Line {line_no}: JSON decode error: {exc}")

    print(f"\n[1] Structural Integrity Check:")
    print(f"  • Total lines parsed: {len(records)}")
    if len(records) != 100:
        errors.append(f"Expected exactly 100 scenarios, found {len(records)}")
    else:
        print(f"  • Scenario count check: PASS (Exactly 100 scenarios)")

    # 2. Schema and Content Verification
    seen_ids = set()
    category_counts = {}
    region_counts = {}
    city_counts = {}
    language_counts = {}
    action_counts = {}

    pref_extraction_count = 0
    dialogue_count = 0
    location_agnostic_count = 0
    budget_clearing_count = 0
    unlimited_budget_count = 0
    accessibility_count = 0
    dietary_count = 0
    safety_count = 0

    user_prompts = []

    for line_no, rec in records:
        scenario_id = rec.get("id")
        if not scenario_id:
            errors.append(f"Line {line_no}: missing 'id'")
        elif scenario_id in seen_ids:
            errors.append(f"Line {line_no}: duplicate ID '{scenario_id}'")
        seen_ids.add(scenario_id)

        cat = rec.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

        reg = rec.get("region", "unknown")
        region_counts[reg] = region_counts.get(reg, 0) + 1

        city = rec.get("city", "unknown")
        city_counts[city] = city_counts.get(city, 0) + 1

        lang = rec.get("language_style", "unknown")
        language_counts[lang] = language_counts.get(lang, 0) + 1

        action = rec.get("expected_action", "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1

        messages = rec.get("messages", [])
        if not messages or not isinstance(messages, list):
            errors.append(f"Scenario {scenario_id}: 'messages' must be a non-empty list")
            continue

        roles = [m.get("role") for m in messages if isinstance(m, dict)]
        if "user" not in roles:
            errors.append(f"Scenario {scenario_id}: missing user turn")
        if roles and roles[-1] != "assistant":
            errors.append(f"Scenario {scenario_id}: final message is not assistant (got {roles[-1]})")

        user_text = get_conversation_user_text(messages)
        user_prompts.append((scenario_id, user_text))

        system_text = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        assistant_text = get_final_assistant_text(messages)

        # Feature flags
        if "preference extractor" in system_text:
            pref_extraction_count += 1
            # Check JSON extraction block
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", assistant_text, re.DOTALL)
            if not json_match:
                errors.append(f"Scenario {scenario_id}: missing ```json code block in extraction response")
            else:
                try:
                    parsed_json = json.loads(json_match.group(1))
                    if parsed_json.get("action") != "save_preferences":
                        errors.append(f"Scenario {scenario_id}: action must be 'save_preferences'")
                    extra_keys = set(parsed_json.keys()) - CANONICAL_PREFERENCE_KEYS
                    if extra_keys:
                        errors.append(f"Scenario {scenario_id}: non-canonical root keys {list(extra_keys)}")

                    # Check expected output match
                    exp_out = rec.get("expected_output")
                    if exp_out:
                        for k, v in exp_out.items():
                            if k not in parsed_json:
                                errors.append(f"Scenario {scenario_id}: key '{k}' missing from parsed JSON")
                            elif parsed_json[k] != v:
                                errors.append(f"Scenario {scenario_id}: value mismatch for '{k}': expected {v!r}, got {parsed_json[k]!r}")
                except Exception as exc:
                    errors.append(f"Scenario {scenario_id}: failed to parse JSON in code block: {exc}")
        else:
            dialogue_count += 1

        # Check domain specific flags
        if cat == "location_agnostic" or (rec.get("city") == "None" and "save_preferences" in system_text):
            location_agnostic_count += 1
        if "clear" in user_text.lower() and "budget" in user_text.lower():
            budget_clearing_count += 1
        if "unlimited" in user_text.lower() and "budget" in user_text.lower():
            unlimited_budget_count += 1
        if cat == "accessibility_inclusivity":
            accessibility_count += 1
        if cat == "dietary_inclusivity":
            dietary_count += 1
        if cat == "safety_and_boundaries":
            safety_count += 1

    print(f"  • Preference Extraction Scenarios: {pref_extraction_count}")
    print(f"  • Conversational & Planning Scenarios: {dialogue_count}")
    print(f"  • Unique Scenario IDs: {len(seen_ids)} / 100")

    # 3. Category & Geographic Distribution Reporting
    print(f"\n[2] Category Distribution (10 Scenarios per Category):")
    for cat, count in sorted(category_counts.items()):
        status = "PASS" if count == 10 else "FAIL"
        print(f"  • {cat:<32}: {count:>2} [{status}]")
        if count != 10:
            errors.append(f"Category '{cat}' has {count} scenarios, expected 10")

    print(f"\n[3] Regional Geographic Representation:")
    for reg, count in sorted(region_counts.items(), key=lambda x: -x[1]):
        print(f"  • {reg:<35}: {count:>2} ({count/100*100:.1f}%)")

    print(f"\n[4] City Representation Breakdown:")
    for city, count in sorted(city_counts.items(), key=lambda x: -x[1]):
        print(f"  • {city:<35}: {count:>2} ({count/100*100:.1f}%)")

    print(f"\n[5] Language & Linguistic Style Breakdown:")
    for lang, count in sorted(language_counts.items(), key=lambda x: -x[1]):
        print(f"  • {lang:<35}: {count:>2} ({count/100*100:.1f}%)")

    # 4. Internal Duplication Checks
    print(f"\n[6] Internal Duplicate & Near-Duplicate Check:")
    exact_internal_dups = 0
    near_internal_dups = 0
    for i in range(len(user_prompts)):
        id_i, prompt_i = user_prompts[i]
        for j in range(i + 1, len(user_prompts)):
            id_j, prompt_j = user_prompts[j]
            if prompt_i.strip().lower() == prompt_j.strip().lower():
                errors.append(f"Exact internal duplicate prompt between {id_i} and {id_j}")
                exact_internal_dups += 1
            else:
                sim = jaccard_similarity(prompt_i, prompt_j)
                if sim > 0.85:
                    warnings.append(f"High Jaccard similarity ({sim:.2f}) between {id_i} and {id_j}")
                    near_internal_dups += 1

    print(f"  • Exact internal duplicate prompts: {exact_internal_dups}")
    print(f"  • High-similarity pairs (>0.85 Jaccard): {near_internal_dups}")

    # 5. Cross-Dataset Leakage Audit
    print(f"\n[7] Cross-Dataset Overlap & Leakage Audit:")
    datasets_to_check = [
        ("data/train_expanded_v2.jsonl", DATA_DIR / "train_expanded_v2.jsonl"),
        ("data/train.jsonl", DATA_DIR / "train.jsonl"),
        ("data/eval.jsonl", DATA_DIR / "eval.jsonl"),
        ("data/eval_expanded.jsonl", DATA_DIR / "eval_expanded.jsonl"),
    ]

    leakage_detected = 0
    for name, path in datasets_to_check:
        if not path.exists():
            print(f"  • {name}: File not found, skipping")
            continue

        existing_prompts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        p = get_conversation_user_text(rec.get("messages", []))
                        if p:
                            existing_prompts.append(p)
                    except:
                        pass

        exact_matches = 0
        max_sim = 0.0
        max_pair = ("", "")
        for pan_id, pan_prompt in user_prompts:
            pan_norm = pan_prompt.strip().lower()
            for ex_prompt in existing_prompts:
                ex_norm = ex_prompt.strip().lower()
                if pan_norm == ex_norm:
                    exact_matches += 1
                    leakage_detected += 1
                    errors.append(f"LEAKAGE: Exact duplicate between {pan_id} and {name}: '{pan_prompt}'")
                sim = jaccard_similarity(pan_norm, ex_norm)
                if sim > max_sim:
                    max_sim = sim
                    max_pair = (pan_prompt, ex_prompt)

        status = "PASS (0 exact matches)" if exact_matches == 0 else f"FAIL ({exact_matches} exact matches)"
        print(f"  • {name:<30}: {status} (Max Jaccard similarity: {max_sim:.3f})")

    # 6. Evaluation Harness (evaluate.py) Compatibility Simulation
    print(f"\n[8] Evaluation Harness Compatibility Analysis:")
    try:
        from evaluate import evaluate_response
        harness_passes = 0
        harness_violations = 0
        harness_notes = []

        for line_no, rec in records:
            scenario_id = rec.get("id")
            messages = rec.get("messages", [])
            user_text = get_conversation_user_text(messages)
            system_text = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
            ref_resp = get_final_assistant_text(messages)

            eval_res = evaluate_response(ref_resp, user_prompt=user_text, system_prompt=system_text)
            if eval_res["is_safe"]:
                harness_passes += 1
            else:
                harness_violations += 1
                harness_notes.append((scenario_id, eval_res["violations"]))

        print(f"  • Reference completions passed by evaluate.py: {harness_passes} / 100 ({harness_passes/100*100:.1f}%)")
        if harness_violations > 0:
            print(f"  • Reference completions flagged by evaluate.py: {harness_violations}")
            for sid, viols in harness_notes[:8]:
                print(f"    - {sid}: {' | '.join(viols)}")
            if len(harness_notes) > 8:
                print(f"    ... and {len(harness_notes) - 8} more.")
    except Exception as exc:
        print(f"  • Could not run evaluate.py simulation: {exc}")

    # Summary
    print("\n" + "=" * 70)
    print("Audit Summary:")
    print(f"Total Errors:   {len(errors)}")
    print(f"Total Warnings: {len(warnings)}")
    if errors:
        print("\nERRORS ENCOUNTERED:")
        for err in errors[:15]:
            print(f"  [X] {err}")
    else:
        print("\n[ALL BENCHMARK INTEGRITY & LEAKAGE CHECKS PASSED]")
    print("=" * 70)

    return len(errors) == 0


if __name__ == "__main__":
    success = run_benchmark_audit()
    sys.exit(0 if success else 1)
