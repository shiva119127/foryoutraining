#!/usr/bin/env python3
"""
Independent Audit and Quality Verification of Repaired Dataset (v2).
Phase 6C: Comprehensive verification of diversity, grammar, context consistency,
zero leakage, manifest synchronization, and immutability.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import sys
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
V1_TRAIN = DATA_DIR / "train_expanded.jsonl"
V1_MANIFEST = DATA_DIR / "dataset_manifest.jsonl"
V2_TRAIN = DATA_DIR / "train_expanded_v2.jsonl"
V2_MANIFEST = DATA_DIR / "dataset_manifest_v2.jsonl"
EVAL_FILE = DATA_DIR / "eval.jsonl"
EVAL_EXPANDED_FILE = DATA_DIR / "eval_expanded.jsonl"
TRAIN_BASELINE = DATA_DIR / "train.jsonl"

ADAPTER_DIR = ROOT / "outputs" / "flowersforyou-adapter"
BACKUP_DIR = ROOT / "outputs" / "phase2-adapter-backup"


def get_md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def tokenize(text: str) -> set:
    return set(re.findall(r"\w+", text.lower()))


def jaccard(s1: str, s2: str) -> float:
    t1 = tokenize(s1)
    t2 = tokenize(s2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def get_user_text(messages: list) -> str:
    return " ".join(m.get("content", "") for m in messages if m.get("role") == "user")


def get_assistant_text(messages: list) -> str:
    return " ".join(m.get("content", "") for m in messages if m.get("role") == "assistant")


def main():
    print("=" * 70)
    print("FLOWERSFORYOU PHASE 6C: INDEPENDENT DATASET V2 QUALITY AUDIT")
    print("=" * 70)

    # 1. Baseline Immutability Check
    print("\n--- 1. Baseline & Artifact Immutability Verification ---")
    expected_hashes = {
        EVAL_FILE: "83221b5d36c71b4b50d7439541678d16",
        EVAL_EXPANDED_FILE: "46c5be6cab32ec808df7ef3c6f13ddaa",
        TRAIN_BASELINE: "4edb9c60fe6dd3dfa6d65e40d34b6e69",
        V1_TRAIN: "7f584fe28d2986d560cdcaf15e4c5ed2",
    }
    all_immut_pass = True
    for p, exp_h in expected_hashes.items():
        actual_h = get_md5(p)
        status = "MATCH" if actual_h == exp_h else "MISMATCH"
        if status != "MATCH":
            all_immut_pass = False
        print(f"[{status}] {p.name}: {actual_h}")
    print(f"Baseline files immutability: {'PASSED' if all_immut_pass else 'FAILED'}")

    # Adapter directory checks
    adapter_weights = ADAPTER_DIR / "adapter_model.safetensors"
    backup_weights = BACKUP_DIR / "adapter_model.safetensors"
    print(f"Active Adapter weights: {adapter_weights.stat().st_size:,} bytes")
    print(f"Backup Adapter weights: {backup_weights.stat().st_size:,} bytes")

    # 2. Volume and Schema Verification
    print("\n--- 2. Dataset Volume and Schema Verification ---")
    v2_records = []
    with open(V2_TRAIN, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            row = json.loads(line)
            v2_records.append(row)

    v2_manifest = []
    with open(V2_MANIFEST, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            row = json.loads(line)
            v2_manifest.append(row)

    print(f"Total v2 Training Records: {len(v2_records)}")
    print(f"Total v2 Manifest Records: {len(v2_manifest)}")
    assert len(v2_records) == len(v2_manifest) == 10000, "Count mismatch!"

    # 3. Exact and Normalized Duplicates
    print("\n--- 3. Exact and Normalized Duplication Analysis ---")
    exact_user_prompts = set()
    exact_assistant_resps = set()
    exact_full_conversations = set()
    norm_user_prompts = set()

    exact_u_dups = 0
    exact_conv_dups = 0
    norm_u_dups = 0

    for r in v2_records:
        msgs = r.get("messages", [])
        u_text = get_user_text(msgs)
        a_text = get_assistant_text(msgs)
        conv_str = json.dumps(msgs, ensure_ascii=False)
        norm_u = " ".join(u_text.lower().split())

        if u_text in exact_user_prompts:
            exact_u_dups += 1
        exact_user_prompts.add(u_text)

        if conv_str in exact_full_conversations:
            exact_conv_dups += 1
        exact_full_conversations.add(conv_str)

        if norm_u in norm_user_prompts:
            norm_u_dups += 1
        norm_user_prompts.add(norm_u)

    print(f"Exact Full Conversation Duplicates: {exact_conv_dups}")
    print(f"Exact User Prompt Duplicates:       {exact_u_dups}")
    print(f"Normalized User Prompt Duplicates:  {norm_u_dups}")

    # 4. Global Near-Duplicate Analysis across all 10,000 records
    print("\n--- 4. Global Near-Duplicate Token Analysis (All 10,000 Records) ---")
    toks_list = [tokenize(get_user_text(r["messages"])) for r in v2_records]
    
    # Fast length-partitioned index to check all 49,995,000 pairs
    by_len = defaultdict(list)
    near_dups_85 = 0
    near_dups_80 = 0
    near_dups_75 = 0
    max_jaccard_internal = 0.0
    highest_pair = None

    for i, t1 in enumerate(toks_list):
        l1 = len(t1)
        if l1 == 0:
            continue
        min_l2 = int(l1 * 0.70)
        max_l2 = int(l1 / 0.70)

        for l2 in range(min_l2, max_l2 + 1):
            for j, t2 in by_len[l2]:
                inter = len(t1 & t2)
                sim = inter / (l1 + l2 - inter)
                if sim > max_jaccard_internal:
                    max_jaccard_internal = sim
                    highest_pair = (j, i, sim)
                if sim >= 0.85:
                    near_dups_85 += 1
                if sim >= 0.80:
                    near_dups_80 += 1
                if sim >= 0.75:
                    near_dups_75 += 1

        by_len[l1].append((i, t1))

    print(f"Max Internal User Prompt Jaccard: {max_jaccard_internal:.4f}")
    if highest_pair:
        j_idx, i_idx, s = highest_pair
        print(f"Highest Pair Match (similarity={s:.4f}):")
        print(f"  Record #{j_idx+1}: '{get_user_text(v2_records[j_idx]['messages'])}'")
        print(f"  Record #{i_idx+1}: '{get_user_text(v2_records[i_idx]['messages'])}'")
    print(f"Global Near-Duplicate Pairs (Jaccard >= 0.85): {near_dups_85} (v1 had 8,973)")
    print(f"Global Near-Duplicate Pairs (Jaccard >= 0.80): {near_dups_80}")
    print(f"Global Near-Duplicate Pairs (Jaccard >= 0.75): {near_dups_75}")

    # 5. Held-Out Evaluation Leakage Analysis
    print("\n--- 5. Held-Out Evaluation Leakage Analysis ---")
    eval_prompts = []
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            eval_prompts.append(("eval_baseline", get_user_text(r["messages"])))
    with open(EVAL_EXPANDED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            eval_prompts.append(("eval_expanded", get_user_text(r["messages"])))

    max_eval_j = 0.0
    max_eval_pair = None
    leakage_count = 0

    for i, r in enumerate(v2_records):
        u = get_user_text(r["messages"])
        for ev_src, ev_u in eval_prompts:
            sim = jaccard(u, ev_u)
            if sim > max_eval_j:
                max_eval_j = sim
                max_eval_pair = (i, ev_src, ev_u, u, sim)
            if sim >= 0.38:
                leakage_count += 1

    print(f"Max Train-to-Eval Jaccard Similarity: {max_eval_j:.4f} (Safety Limit: < 0.40)")
    if max_eval_pair:
        i_idx, ev_src, ev_u, u, sim = max_eval_pair
        print(f"Highest Eval Match (similarity={sim:.4f}):")
        print(f"  Dataset v2 #{i_idx+1}: '{u}'")
        print(f"  Held-Out ({ev_src}): '{ev_u}'")
    print(f"Eval Leakage Violations (Jaccard >= 0.38): {leakage_count}")

    # 6. Category Distribution Comparison (v1 vs v2)
    print("\n--- 6. Category Distribution Comparison (v1 vs v2) ---")
    v1_cats = Counter()
    with open(V1_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            v1_cats[json.loads(line).get("category", "")] += 1

    v2_cats = Counter()
    for m in v2_manifest:
        v2_cats[m.get("category", "")] += 1

    print(f"{'Category':<36} | {'v1 Count':<10} | {'v2 Count':<10} | {'Delta':<6}")
    print("-" * 70)
    for cat in sorted(v1_cats.keys()):
        c1 = v1_cats[cat]
        c2 = v2_cats[cat]
        print(f"{cat:<36} | {c1:<10} | {c2:<10} | {c2 - c1:+d}")

    # 7. Grammar Audit for Plural/Gerund Activities
    print("\n--- 7. Grammar Audit (Plural/Gerund Phrasing Artifacts) ---")
    flawed_grammar_patterns = [
        r"\ba\s+leather\s+craft\s+basics(?!\s+(workshop|class|session|course|studio))\b",
        r"\ba\s+vintage\s+arcade\s+games(?!\s+(session|night|evening|tournament|lounge))\b",
        r"\ba\s+modern\s+strategy\s+board\s+games(?!\s+(session|night|evening|meetup|club))\b",
        r"\ba\s+indie\s+card\s+games(?!\s+(session|night|evening|meetup|club))\b",
        r"\ba\s+botanical\s+sketching(?!\s+(walk|workshop|class|session))\b",
        r"\ba\s+meditative\s+journaling(?!\s+(workshop|class|session|circle))\b",
        r"\ba\s+mindful\s+outdoor\s+yoga(?!\s+(class|session|workshop))\b",
        r"\ba\s+specialty\s+coffee\s+tasting(?!\s+(session|workshop|class|flight))\b",
        r"\ba\s+pasta\s+making\s+from\s+scratch(?!\s+(masterclass|workshop|class))\b",
    ]
    grammar_violations = 0
    for i, r in enumerate(v2_records):
        full_text = " ".join(m.get("content", "") for m in r.get("messages", []))
        for pat in flawed_grammar_patterns:
            if re.search(pat, full_text, re.IGNORECASE):
                grammar_violations += 1
                print(f"Grammar violation in Record #{i+1}: pattern '{pat}' found.")
    print(f"Grammar Violations in v2: {grammar_violations} (Zero tolerance)")

    # 8. Skip-Onboarding Context Amnesia Audit
    print("\n--- 8. Skip-Onboarding Context Amnesia Audit ---")
    skip_amnesia_violations = 0
    skip_records_with_area = 0
    for i, r in enumerate(v2_records):
        m_entry = v2_manifest[i]
        if m_entry.get("category") == "undecided_and_skip_onboarding":
            msgs = r.get("messages", [])
            u = get_user_text(msgs)
            a = get_assistant_text(msgs)
            # Check if user mentioned an area
            for city, areas in [("Bengaluru", ["Indiranagar", "Koramangala", "HSR Layout"]),
                                ("Mumbai", ["Bandra", "Lower Parel", "Colaba"]),
                                ("Pune", ["Baner", "Koregaon Park", "Aundh"])]:
                for ar in areas:
                    if ar.lower() in u.lower():
                        skip_records_with_area += 1
                        # If assistant asks "which area" or "which neighborhood", that is amnesia!
                        if re.search(r"which (area|neighborhood)", a, re.IGNORECASE):
                            skip_amnesia_violations += 1
                            print(f"Context amnesia in Record #{i+1}: User stated {ar}, assistant asked: '{a}'")
    print(f"Audited skip-onboarding with pre-stated area: {skip_records_with_area} instances.")
    print(f"Skip-Onboarding Context Amnesia Violations in v2: {skip_amnesia_violations}")

    # 9. Structured Preference Extraction Canonical Schema Audit
    print("\n--- 9. Preference Extraction Canonical Schema Audit ---")
    schema_failures = 0
    pref_count = 0
    for i, r in enumerate(v2_records):
        m_entry = v2_manifest[i]
        if m_entry.get("category") == "structured_preference_extraction":
            pref_count += 1
            a = get_assistant_text(r.get("messages", []))
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", a, re.DOTALL)
            if not match:
                schema_failures += 1
                continue
            try:
                data = json.loads(match.group(1))
                if data.get("action") != "save_preferences":
                    schema_failures += 1
                if not isinstance(data.get("interests"), list):
                    schema_failures += 1
                if not isinstance(data.get("preferred_neighborhoods"), list):
                    schema_failures += 1
                if not isinstance(data.get("budget_inr_max"), (int, float)):
                    schema_failures += 1
            except Exception:
                schema_failures += 1
    print(f"Audited structured preference extraction records: {pref_count}")
    print(f"Preference Extraction Schema Failures in v2: {schema_failures}")

    # 10. Summary Report
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY:")
    print(f"  Total records:                       {len(v2_records)}")
    print(f"  Exact duplicates:                    {exact_conv_dups}")
    print(f"  Near-duplicate pairs (J >= 0.85):    {near_dups_85}")
    print(f"  Near-duplicate pairs (J >= 0.80):    {near_dups_80}")
    print(f"  Max Internal User Jaccard:           {max_jaccard_internal:.4f}")
    print(f"  Max Eval Jaccard:                    {max_eval_j:.4f}")
    print(f"  Eval Leakage count:                  {leakage_count}")
    print(f"  Grammar violations:                  {grammar_violations}")
    print(f"  Context amnesia violations:          {skip_amnesia_violations}")
    print(f"  Canonical preference schema errors:  {schema_failures}")
    print(f"  Baseline files immutable:            {'YES' if all_immut_pass else 'NO'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
