#!/usr/bin/env python3
"""
FlowersForYou - End-to-End Application Integration Test Fixture.

Validates the full application integration path:
1. Model inference execution (Phase 6D adapter).
2. Parsing & schema validation of structured output.
3. Schema translation bridge (Action: 'save_preferences' -> Application StructuredMemoryRecord).
4. Isolated database persistence (reproducing MemoryManager.reconcileAndSave).
5. Profile retrieval (reproducing MemoryManager.getActiveMemorySummary).
6. Memory injection into Planner prompt.
7. Preference correction / conflict resolution (verifying newer explicit values supersede older ones).
8. Rejection of invalid model outputs without database corruption.
"""

import json
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "outputs/flowersforyou-adapter-10k"


# =====================================================================
# Isolated In-Memory Test Database (Faithful reproduction of db & MemoryManager)
# =====================================================================
class IsolatedTestDatabase:
    def __init__(self):
        self.users = {
            "test_user_aarav": {
                "id": "test_user_aarav",
                "name": "Aarav Test",
                "raw_records": [],
                "temporary_context": [],
            }
        }

    def get_user_memory(self, user_id: str):
        user = self.users.get(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        return deepcopy(user)

    def save_user_memory(self, user_id: str, records: list, temp_context: list = None):
        if user_id not in self.users:
            self.users[user_id] = {"id": user_id, "raw_records": [], "temporary_context": []}
        self.users[user_id]["raw_records"] = deepcopy(records)
        if temp_context is not None:
            self.users[user_id]["temporary_context"] = deepcopy(temp_context)


class ApplicationMemoryManager:
    def __init__(self, db: IsolatedTestDatabase):
        self.db = db

    def reconcile_and_save(self, user_id: str, incoming_records: list, temp_context: list = None):
        user_mem = self.db.get_user_memory(user_id)
        existing = user_mem.get("raw_records", [])
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        updated_records = deepcopy(existing)

        for inc in incoming_records:
            match_idx = None
            for idx, ex in enumerate(updated_records):
                if ex["category"] == inc["category"] and ex["key"] == inc["key"] and ex["status"] == "active":
                    match_idx = idx
                    break

            if match_idx is not None:
                ex = updated_records[match_idx]
                if str(ex["value"]).lower() == str(inc["value"]).lower():
                    # Idempotent re-confirmation
                    ex["last_confirmed_at"] = now
                    ex["confidence"] = max(ex.get("confidence", 0.8), inc.get("confidence", 0.9))
                else:
                    # Conflict: Newer explicit statement supersedes
                    ex["status"] = "superseded"
                    ex["superseded_at"] = now
                    new_rec = deepcopy(inc)
                    new_rec["status"] = "active"
                    new_rec["created_at"] = now
                    new_rec["last_confirmed_at"] = now
                    updated_records.append(new_rec)
            else:
                new_rec = deepcopy(inc)
                new_rec["status"] = "active"
                new_rec["created_at"] = now
                new_rec["last_confirmed_at"] = now
                updated_records.append(new_rec)

        self.db.save_user_memory(user_id, updated_records, temp_context)

    def get_active_memory_summary(self, user_id: str):
        user_mem = self.db.get_user_memory(user_id)
        active = [r for r in user_mem.get("raw_records", []) if r.get("status") == "active"]

        interests = [r["value"] for r in active if r["category"] == "interest"]
        neighborhoods = [r["value"] for r in active if r["category"] == "location"]
        budgets = [r["value"] for r in active if r["category"] == "budget"]

        return {
            "interests": interests,
            "preferred_neighborhoods": neighborhoods,
            "budget_max": budgets[-1] if budgets else None,
            "total_active_records": len(active),
            "raw_active": active,
        }


# =====================================================================
# Application Schema Validator & Bridge
# =====================================================================
class ApplicationSchemaBridge:
    """Translates the fine-tuned model's 'save_preferences' output into Application Memory Records."""

    @staticmethod
    def validate_and_translate(raw_json_str: str, user_id: str):
        try:
            data = json.loads(raw_json_str)
        except Exception as exc:
            raise ValueError(f"JSON Syntax Error: {exc}")

        if not isinstance(data, dict):
            raise ValueError("Preference extraction payload is not an object")

        if data.get("action") != "save_preferences":
            raise ValueError(f"Expected action: 'save_preferences', got {repr(data.get('action'))}")

        records = []
        # Translate interests
        for item in data.get("interests", []):
            slug = re.sub(r"\W+", "_", str(item).lower()).strip("_")
            records.append({
                "user_id": user_id,
                "category": "interest",
                "key": slug,
                "value": str(item).replace("_", " ").title(),
                "source": "explicit",
                "confidence": 0.95,
            })

        # Translate preferred neighborhoods
        for hood in data.get("preferred_neighborhoods", []):
            slug = re.sub(r"\W+", "_", str(hood).lower()).strip("_")
            records.append({
                "user_id": user_id,
                "category": "location",
                "key": f"neighborhood_{slug}",
                "value": str(hood),
                "source": "explicit",
                "confidence": 0.95,
            })

        # Translate budget
        if data.get("budget_inr_max") is not None:
            b_val = data["budget_inr_max"]
            if not isinstance(b_val, (int, float)):
                raise ValueError(f"Budget must be numeric, got {type(b_val)}")
            records.append({
                "user_id": user_id,
                "category": "budget",
                "key": "budget_ceiling_inr",
                "value": int(b_val),
                "source": "explicit",
                "confidence": 0.95,
            })

        return records


# =====================================================================
# Main Integration Test Execution
# =====================================================================
def run_integration_tests():
    print("=" * 70)
    print("FLOWERSFORYOU PHASE 6D — APPLICATION INTEGRATION TEST SUITE")
    print("=" * 70)

    # 1. Initialize isolated database and manager
    db = IsolatedTestDatabase()
    mem_mgr = ApplicationMemoryManager(db)
    test_user_id = "test_user_aarav"

    # 2. Load model
    print(f"\n[1/6] Loading Phase 6D Adapter from {ADAPTER_PATH}...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "mps" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(device)
    model.eval()

    def query_model(messages, max_tokens=120):
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

    test_results = []

    # -------------------------------------------------------------
    # Step A: User states initial preferences -> Model extracts JSON
    # -------------------------------------------------------------
    print("\n[2/6] Testing Initial Preference Extraction & Persistence...")
    user_msg_1 = "Please save my profile: my interests are specialty coffee tasting and linocut printmaking, budget ceiling is ₹800, prefer Indiranagar."
    messages_1 = [
        {"role": "system", "content": "You are the FlowersForYou preference extractor. Extract stated preferences as valid JSON and do not invent details."},
        {"role": "user", "content": user_msg_1},
    ]
    resp_1 = query_model(messages_1)
    print("Model Output 1:\n", resp_1)

    json_match_1 = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", resp_1, re.DOTALL)
    assert json_match_1 is not None, "Failed to find JSON block in extraction 1"
    raw_json_1 = json_match_1.group(1)

    # Validate and translate via bridge
    records_1 = ApplicationSchemaBridge.validate_and_translate(raw_json_1, test_user_id)
    print(f"Bridge translated {len(records_1)} structured memory records:")
    for r in records_1:
        print(f"  + [{r['category'].upper()}] {r['key']}: {r['value']}")

    # Save to isolated DB
    mem_mgr.reconcile_and_save(test_user_id, records_1)

    # Retrieve and assert
    summary_1 = mem_mgr.get_active_memory_summary(test_user_id)
    print("\nActive Memory Summary 1:", json.dumps(summary_1, indent=2, default=str))

    check_1 = (
        len(summary_1["interests"]) == 2
        and "Indiranagar" in summary_1["preferred_neighborhoods"]
        and summary_1["budget_max"] == 800
    )
    test_results.append({
        "step": "Initial Extraction & Persistence",
        "passed": check_1,
        "detail": f"Active interests: {summary_1['interests']}, Hoods: {summary_1['preferred_neighborhoods']}, Budget: {summary_1['budget_max']}",
    })
    print(f"Step Result: {'[PASS]' if check_1 else '[FAIL]'}")

    # -------------------------------------------------------------
    # Step B: Supplying retrieved profile to Planner
    # -------------------------------------------------------------
    print("\n[3/6] Testing Profile Supply to Planner...")
    planner_sys = (
        f"You are the FlowersForYou planner. The user has active preferences: "
        f"Interests: {', '.join(summary_1['interests'])}; "
        f"Area: {', '.join(summary_1['preferred_neighborhoods'])}; "
        f"Budget: ₹{summary_1['budget_max']}. Respect these constraints."
    )
    user_planner_msg = "What should we do this Saturday afternoon?"
    planner_messages = [
        {"role": "system", "content": planner_sys},
        {"role": "user", "content": user_planner_msg},
    ]
    planner_resp = query_model(planner_messages)
    print("Planner Output:\n", planner_resp)

    check_2 = (
        any(k in planner_resp.lower() for k in ["indiranagar", "coffee", "linocut", "printmaking", "craft", "800", "budget"])
        and "?" in planner_resp
    )
    test_results.append({
        "step": "Supplying Profile to Planner",
        "passed": check_2,
        "detail": f"Planner response aligned with stored profile constraints (mentions relevant terms & asks question).",
    })
    print(f"Step Result: {'[PASS]' if check_2 else '[FAIL]'}")

    # -------------------------------------------------------------
    # Step C: Returning user corrects preference (Conflict Resolution)
    # -------------------------------------------------------------
    print("\n[4/6] Testing Preference Update & Conflict Resolution...")
    user_msg_corr = "Update my spending limit: please raise my maximum budget to ₹1500, and change my area to Koramangala."
    messages_corr = [
        {"role": "system", "content": "You are the FlowersForYou preference extractor. Extract stated preferences as valid JSON and do not invent details."},
        {"role": "user", "content": user_msg_corr},
    ]
    resp_corr = query_model(messages_corr)
    print("Model Output Correction:\n", resp_corr)

    json_match_corr = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", resp_corr, re.DOTALL)
    assert json_match_corr is not None, "Failed to find JSON block in correction"
    records_corr = ApplicationSchemaBridge.validate_and_translate(json_match_corr.group(1), test_user_id)

    # Reconcile update into DB
    mem_mgr.reconcile_and_save(test_user_id, records_corr)

    summary_updated = mem_mgr.get_active_memory_summary(test_user_id)
    print("\nUpdated Active Memory Summary:", json.dumps(summary_updated, indent=2, default=str))

    # Verify conflict resolution:
    # 1. Budget updated from 800 to 1500
    # 2. Old budget record is marked 'superseded' in raw records
    raw_user = db.get_user_memory(test_user_id)["raw_records"]
    budget_records = [r for r in raw_user if r["category"] == "budget"]
    superseded_budgets = [r for r in budget_records if r["status"] == "superseded"]
    active_budgets = [r for r in budget_records if r["status"] == "active"]

    check_3 = (
        summary_updated["budget_max"] == 1500
        and len(active_budgets) == 1
        and len(superseded_budgets) >= 1
        and superseded_budgets[0]["value"] == 800
    )
    test_results.append({
        "step": "Preference Update & Conflict Resolution",
        "passed": check_3,
        "detail": f"Active budget: {summary_updated['budget_max']} (superseded old budget 800 successfully).",
    })
    print(f"Step Result: {'[PASS]' if check_3 else '[FAIL]'}")

    # -------------------------------------------------------------
    # Step D: Rejecting Invalid Model Output Safely
    # -------------------------------------------------------------
    print("\n[5/6] Testing Safety Rejection of Malformed / Invalid Output...")
    malformed_outputs = [
        '{"preferences": {"budget": [₹700]}}',  # Phase 4 schema failure (invalid root keys & invalid syntax)
        '{"action": "wrong_action", "interests": ["cycling"]}',  # Invalid action dispatch
        '{"action": "save_preferences", "budget_inr_max": "cheap"}',  # Non-numeric budget
        'Not a JSON block at all',  # Complete non-JSON output
    ]

    rejected_count = 0
    db_records_before = len(db.get_user_memory(test_user_id)["raw_records"])

    for malformed in malformed_outputs:
        try:
            ApplicationSchemaBridge.validate_and_translate(malformed, test_user_id)
            print(f"  [ERROR] Malformed output was unexpectedly accepted: {malformed}")
        except Exception as err:
            rejected_count += 1
            print(f"  [OK] Safely caught invalid output: {err}")

    db_records_after = len(db.get_user_memory(test_user_id)["raw_records"])
    check_4 = (rejected_count == len(malformed_outputs)) and (db_records_before == db_records_after)

    test_results.append({
        "step": "Safe Rejection of Invalid Output",
        "passed": check_4,
        "detail": f"All {rejected_count}/{len(malformed_outputs)} malformed payloads safely rejected; zero DB corruption.",
    })
    print(f"Step Result: {'[PASS]' if check_4 else '[FAIL]'}")

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("APPLICATION INTEGRATION TEST SUMMARY")
    print("=" * 70)
    all_passed = all(r["passed"] for r in test_results)
    for r in test_results:
        print(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['step']}: {r['detail']}")
    print(f"\nOverall Result: {'PASSED' if all_passed else 'FAILED'}")

    out_file = Path("scratch/integration_test_results.json")
    out_file.write_text(json.dumps(test_results, indent=2), encoding="utf-8")
    print(f"Saved results to {out_file}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_integration_tests())
