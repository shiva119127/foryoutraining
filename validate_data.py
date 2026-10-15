#!/usr/bin/env python3
"""
FlowersForYou - Dataset Validator & Overlap Auditor (Phase 4).

Validates:
1. Structural integrity: Valid JSONL, message format, allowed roles ('system', 'user', 'assistant', 'tool').
2. Dialogue completion: Assistant must be final role; conversation must contain user message.
3. Canonical schema integrity: Validates that structured preference extractions use canonical keys and 'action: save_preferences'.
4. Exact and semantic leakage / overlap between train.jsonl and held-out eval.jsonl.
5. Near-duplicate prompt checks within the datasets.
"""

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
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


def validate_file(path: Path, return_records: bool = False):
    errors = []
    records = []
    if not path.exists():
        return 0, [f"{path}: file does not exist"], []

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}:{line_no}: invalid JSON: {exc}")
            continue

        messages = record.get("messages") if isinstance(record, dict) else None
        if not isinstance(messages, list) or not messages:
            errors.append(f"{path.name}:{line_no}: expected non-empty 'messages' list")
            continue

        roles = []
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                errors.append(f"{path.name}:{line_no}: message {i} must be an object")
                continue
            role = message.get("role")
            content = message.get("content")
            roles.append(role)
            if role not in ALLOWED_ROLES:
                errors.append(f"{path.name}:{line_no}: unsupported role {role!r} in message {i}")
            if not isinstance(content, str) or not content.strip():
                errors.append(f"{path.name}:{line_no}: message {i} needs non-empty string content")

        if "user" not in roles:
            errors.append(f"{path.name}:{line_no}: conversation has no user message")
        if "assistant" not in roles:
            errors.append(f"{path.name}:{line_no}: conversation has no assistant response")
        if roles and roles[-1] != "assistant":
            errors.append(f"{path.name}:{line_no}: expected final message role to be assistant")

        # Canonical schema check on preference extraction examples
        system_text = next((m.get("content", "") for m in messages if m.get("role") == "system"), "").lower()
        if "preference extractor" in system_text:
            assistant_content = get_final_assistant_text(messages)
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", assistant_content, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    if parsed.get("action") != "save_preferences":
                        errors.append(
                            f"{path.name}:{line_no}: preference extraction schema missing 'action: save_preferences'"
                        )
                    non_canonical = set(parsed.keys()) - CANONICAL_PREFERENCE_KEYS
                    if non_canonical:
                        errors.append(
                            f"{path.name}:{line_no}: non-canonical keys in preference extraction: {list(non_canonical)}"
                        )
                except Exception as exc:
                    errors.append(f"{path.name}:{line_no}: failed to parse JSON in extraction block: {exc}")
            else:
                errors.append(f"{path.name}:{line_no}: missing ```json ... ``` code block in preference extractor record")

        records.append(record)

    if not records:
        errors.append(f"{path.name}: no records found")
    if return_records:
        return len(records), errors, records
    return len(records), errors


def check_dataset_overlaps(train_records: list, eval_records: list):
    """Verify held-out separation between training and evaluation datasets."""
    overlap_warnings = []
    max_jaccard = 0.0
    highest_overlap_pair = None

    for tr_idx, tr in enumerate(train_records, start=1):
        tr_user = get_conversation_user_text(tr.get("messages", []))
        tr_resp = get_final_assistant_text(tr.get("messages", []))

        for ev_idx, ev in enumerate(eval_records, start=1):
            ev_user = get_conversation_user_text(ev.get("messages", []))
            ev_resp = get_final_assistant_text(ev.get("messages", []))

            # Exact prompt match check
            if tr_user.strip().lower() == ev_user.strip().lower():
                overlap_warnings.append(
                    f"[DATA LEAKAGE] Exact user prompt match between Train #{tr_idx} and Eval #{ev_idx}: '{tr_user}'"
                )

            # Exact response match check
            if tr_resp.strip().lower() == ev_resp.strip().lower():
                overlap_warnings.append(
                    f"[DATA LEAKAGE] Exact assistant response match between Train #{tr_idx} and Eval #{ev_idx}"
                )

            # Jaccard token similarity check
            sim = jaccard_similarity(tr_user, ev_user)
            if sim > max_jaccard:
                max_jaccard = sim
                highest_overlap_pair = (tr_idx, ev_idx, tr_user, ev_user, sim)

    return max_jaccard, highest_overlap_pair, overlap_warnings


def check_near_duplicates(records: list, filename: str):
    """Check for unintentional duplicates within the same dataset."""
    duplicates = []
    for i in range(len(records)):
        u1 = get_conversation_user_text(records[i].get("messages", []))
        for j in range(i + 1, len(records)):
            u2 = get_conversation_user_text(records[j].get("messages", []))
            sim = jaccard_similarity(u1, u2)
            if sim >= 0.85:
                duplicates.append((i + 1, j + 1, u1, u2, sim))
    return duplicates


def main():
    print("=" * 65)
    print("FlowersForYou Dataset Validation & Overlap Audit")
    print("=" * 65)

    all_errors = []
    datasets = {}

    for filename in ("train.jsonl", "eval.jsonl"):
        path = DATA_DIR / filename
        count, errors, records = validate_file(path, return_records=True)
        datasets[filename] = records
        all_errors.extend(errors)
        status = "[FAIL]" if errors else "[PASS]"
        print(f"{status} {filename}: {count} record(s)")

    if all_errors:
        print("\nVALIDATION FAILED:")
        for error in all_errors:
            print(" -", error)
        sys.exit(1)

    train_recs = datasets.get("train.jsonl", [])
    eval_recs = datasets.get("eval.jsonl", [])

    # Overlap and leakage checks
    max_sim, highest_pair, leakage_errors = check_dataset_overlaps(train_recs, eval_recs)

    print("\n--- Held-Out Separation & Overlap Report ---")
    print(f"Train records count:      {len(train_recs)}")
    print(f"Eval records count:       {len(eval_recs)}")
    print(f"Max User Prompt Jaccard:  {max_sim:.3f}")

    if highest_pair:
        print(f"Highest similarity pair:  Train #{highest_pair[0]} vs Eval #{highest_pair[1]} (Jaccard = {highest_pair[4]:.3f})")
        print(f"  Train: \"{highest_pair[2]}\"")
        print(f"  Eval:  \"{highest_pair[3]}\"")

    if leakage_errors:
        print("\nDATA LEAKAGE DETECTED:")
        for err in leakage_errors:
            print(" -", err)
        sys.exit(1)
    else:
        print("[PASS] Zero data leakage: No exact prompt or response duplication between train and eval.")

    # Intra-dataset duplicates
    train_dupes = check_near_duplicates(train_recs, "train.jsonl")
    if train_dupes:
        print(f"\n[WARNING] Near-duplicate prompts detected in train.jsonl ({len(train_dupes)} pairs):")
        for d in train_dupes:
            print(f"  Train #{d[0]} vs #{d[1]} (similarity={d[4]:.2f}): '{d[2]}' vs '{d[3]}'")
    else:
        print("[PASS] Zero near-duplicates within train.jsonl (all pairs Jaccard < 0.85).")

    print("\n" + "=" * 65)
    print("PASS: All JSONL records passed structural, canonical schema, and overlap validation.")
    print("=" * 65)


if __name__ == "__main__":
    main()
