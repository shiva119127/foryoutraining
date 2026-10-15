#!/usr/bin/env python3
"""
FlowersForYou - Programmatic Preference Extraction Test Suite for Phase 6D Adapter.

Tests the Phase 6D adapter across 5 core scenarios:
1. Interests and preferred neighborhoods
2. Numeric INR budget
3. Missing preferences (partial inputs without hallucinating omitted fields)
4. Correcting a previously stated preference
5. Multiple preferences in a multi-turn conversation

Validates output against the canonical schema:
{
  "action": "save_preferences",
  "interests": ["..."],
  "preferred_neighborhoods": ["..."],
  "budget_inr_max": 700
}
"""

import json
import re
import sys
import time
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "outputs/flowersforyou-adapter-10k"
SYSTEM_PROMPT = "You are the FlowersForYou preference extractor. Extract stated preferences as valid JSON and do not invent details."

CANONICAL_KEYS = {
    "action",
    "interests",
    "preferred_neighborhoods",
    "budget_inr_max",
    "budget_inr_min",
    "timing",
    "social_vibe",
    "max_group_size",
}


def load_model():
    print(f"Loading Base Model: {MODEL_ID}...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "mps" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    print(f"Attaching PEFT Adapter: {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(device)
    model.eval()
    return model, tokenizer, device


def generate_completion(model, tokenizer, device, messages, max_tokens=150):
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.3,  # lower temp for structured tasks
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def validate_extraction(response_text, expected):
    """Programmatically validates the JSON structure against the canonical schema and expectations."""
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if not json_match:
        return False, "Missing ```json ... ``` code block in response", None

    try:
        data = json.loads(json_match.group(1))
    except Exception as exc:
        return False, f"Invalid JSON syntax: {exc}", None

    if not isinstance(data, dict):
        return False, "Extracted JSON is not an object", data

    # 1. Action check
    if data.get("action") != "save_preferences":
        return False, f"Missing or invalid action key: expected 'save_preferences', got {repr(data.get('action'))}", data

    # 2. Canonical keys only
    extra_keys = set(data.keys()) - CANONICAL_KEYS
    if extra_keys:
        return False, f"Non-canonical keys found: {list(extra_keys)}", data

    # 3. Expected interests check
    if "expected_interests" in expected:
        interests = data.get("interests", [])
        if not isinstance(interests, list):
            return False, f"'interests' must be a list of strings, got {type(interests)}", data
        for exp_int in expected["expected_interests"]:
            if not any(exp_int.lower() in str(i).lower() for i in interests):
                return False, f"Expected interest '{exp_int}' not found in {interests}", data

    # 4. Expected omitted interests
    if expected.get("no_interests") and "interests" in data and len(data["interests"]) > 0:
        return False, f"Expected no interests, but got: {data['interests']}", data

    # 5. Expected neighborhoods check
    if "expected_neighborhoods" in expected:
        hoods = data.get("preferred_neighborhoods", [])
        if not isinstance(hoods, list):
            return False, f"'preferred_neighborhoods' must be a list of strings, got {type(hoods)}", data
        for exp_hood in expected["expected_neighborhoods"]:
            if not any(exp_hood.lower() in str(h).lower() for h in hoods):
                return False, f"Expected neighborhood '{exp_hood}' not found in {hoods}", data

    # 6. Expected omitted neighborhoods
    if expected.get("no_neighborhoods") and "preferred_neighborhoods" in data and len(data["preferred_neighborhoods"]) > 0:
        return False, f"Expected no neighborhoods, but got: {data['preferred_neighborhoods']}", data

    # 7. Expected budget check
    if "expected_budget" in expected:
        budget = data.get("budget_inr_max")
        if budget is None:
            return False, f"Expected budget {expected['expected_budget']} was missing (got None)", data
        if not isinstance(budget, (int, float)):
            return False, f"Budget must be numeric integer/float, got {type(budget)}: {repr(budget)}", data
        if int(budget) != int(expected["expected_budget"]):
            return False, f"Budget mismatch: expected {expected['expected_budget']}, got {budget}", data

    # 8. Expected omitted budget
    if expected.get("no_budget") and data.get("budget_inr_max") is not None:
        return False, f"Expected no budget invented, but got: {data.get('budget_inr_max')}", data

    return True, "PASSED", data


def run_tests():
    model, tokenizer, device = load_model()

    test_scenarios = [
        {
            "id": "PREF-01",
            "name": "Interests & Preferred Neighborhoods",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Please save my profile: my interests are specialty coffee tasting and artisanal pottery, preferred area is Indiranagar.",
                },
            ],
            "expected": {
                "expected_interests": ["specialty_coffee", "pottery"],
                "expected_neighborhoods": ["Indiranagar"],
                "no_budget": True,
            },
        },
        {
            "id": "PREF-02",
            "name": "Numeric INR Budget & Neighborhood",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Update my spending limit: my maximum budget is ₹1500 per person in Bandra West for indie board games.",
                },
            ],
            "expected": {
                "expected_interests": ["board_games"],
                "expected_neighborhoods": ["Bandra West"],
                "expected_budget": 1500,
            },
        },
        {
            "id": "PREF-03",
            "name": "Missing Preferences (No Hallucinated Budget or Location)",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "I really enjoy modern strategy board games, but I haven't decided on any location or budget yet.",
                },
            ],
            "expected": {
                "expected_interests": ["board_games"],
                "no_neighborhoods": True,
                "no_budget": True,
            },
        },
        {
            "id": "PREF-04",
            "name": "Correcting a Previously Stated Preference",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Save my preferences: I like pottery in Whitefield, budget ₹500.",
                },
                {
                    "role": "assistant",
                    "content": '```json\n{\n  "action": "save_preferences",\n  "interests": ["pottery"],\n  "preferred_neighborhoods": ["Whitefield"],\n  "budget_inr_max": 500\n}\n```\nPreferences saved! Recorded pottery in Whitefield with a ₹500 budget.',
                },
                {
                    "role": "user",
                    "content": "Actually, scratch Whitefield—I moved to Koramangala now, and please raise my maximum budget to ₹1200.",
                },
            ],
            "expected": {
                "expected_neighborhoods": ["Koramangala"],
                "expected_budget": 1200,
            },
        },
        {
            "id": "PREF-05",
            "name": "Multiple Preferences in Multi-Turn Conversation",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "I like outdoor cycling and sourdough baking.",
                },
                {
                    "role": "assistant",
                    "content": '```json\n{\n  "action": "save_preferences",\n  "interests": ["outdoor_cycling", "sourdough_baking"]\n}\n```\nNoted outdoor cycling and sourdough baking.',
                },
                {
                    "role": "user",
                    "content": "Also add specialty coffee tasting to my interests, keep it around ₹800 max in Hauz Khas Village.",
                },
            ],
            "expected": {
                "expected_interests": ["coffee", "cycling"],
                "expected_neighborhoods": ["Hauz Khas"],
                "expected_budget": 800,
            },
        },
    ]

    print("\n" + "=" * 70)
    print("RUNNING PROGRAMMATIC PREFERENCE EXTRACTION VALIDATION")
    print("=" * 70)

    results = []
    for sc in test_scenarios:
        print(f"\n--- Scenario {sc['id']}: {sc['name']} ---")
        t0 = time.time()
        response = generate_completion(model, tokenizer, device, sc["messages"])
        elapsed = time.time() - t0

        passed, reason, extracted_data = validate_extraction(response, sc["expected"])
        status_badge = "[PASS]" if passed else "[FAIL]"
        print(f"Status:   {status_badge} ({elapsed:.2f}s)")
        if not passed:
            print(f"Reason:   {reason}")
        print("Response:\n" + response)
        if extracted_data:
            print("Extracted JSON:", json.dumps(extracted_data, indent=2))

        results.append({
            "id": sc["id"],
            "name": sc["name"],
            "passed": passed,
            "reason": reason,
            "response": response,
            "data": extracted_data,
            "elapsed_s": elapsed,
        })

    print("\n" + "=" * 70)
    print("PREFERENCE EXTRACTION TEST RESULTS SUMMARY")
    print("=" * 70)
    total_passed = sum(1 for r in results if r["passed"])
    for r in results:
        badge = "PASS" if r["passed"] else "FAIL"
        print(f"[{badge}] {r['id']}: {r['name']} ({r['elapsed_s']:.2f}s) - {r['reason']}")
    print(f"\nTotal: {total_passed} / {len(results)} Passed ({(total_passed/len(results))*100:.1f}%)")

    # Save results to scratch/pref_extraction_results.json
    out_file = Path("scratch/pref_extraction_results.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved detailed results to {out_file}")
    return 0 if total_passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(run_tests())
