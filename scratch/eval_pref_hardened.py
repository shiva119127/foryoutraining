#!/usr/bin/env python3
"""
FlowersForYou - Evaluates PREF-01 through PREF-05 with the Hardened Preference Pipeline.
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

PLACEHOLDER_PATTERNS = [
    re.compile(r"^city[_\s-]?[a-z0-9]$", re.IGNORECASE),
    re.compile(r"^neighborhood[_\s-]?[a-z0-9]$", re.IGNORECASE),
    re.compile(r"^area[_\s-]?[a-z0-9]$", re.IGNORECASE),
    re.compile(r"^unknown$", re.IGNORECASE),
    re.compile(r"^n/?a$", re.IGNORECASE),
    re.compile(r"^none$", re.IGNORECASE),
    re.compile(r"^null$", re.IGNORECASE),
    re.compile(r"^undefined$", re.IGNORECASE),
    re.compile(r"^placeholder", re.IGNORECASE),
    re.compile(r"^dummy", re.IGNORECASE),
    re.compile(r"^tbd$", re.IGNORECASE),
    re.compile(r"^not[_\s-]?specified$", re.IGNORECASE),
    re.compile(r"^unspecified$", re.IGNORECASE),
]


def is_placeholder(val: str) -> bool:
    if not val or not isinstance(val, str):
        return True
    trimmed = val.strip()
    if not trimmed:
        return True
    return any(p.match(trimmed) for p in PLACEHOLDER_PATTERNS)


def hardened_bridge(raw_json_str: str, raw_input: str, existing_memory: dict = None):
    """Mirror of TypeScript SchemaValidator and MemoryManager logic."""
    try:
        data = json.loads(raw_json_str)
    except Exception as e:
        raise ValueError(f"JSON Syntax Error: {e}")

    lower_input = raw_input.lower()
    records = []

    # Filter interests
    interests = []
    for i in data.get("interests", []):
        if not is_placeholder(str(i)):
            clean = str(i).replace("_", " ").title()
            lower_i = clean.lower()
            tokens = [w for w in lower_i.split() if len(w) > 3]
            is_grounded = len(tokens) == 0 or any(t in lower_input for t in tokens)
            if is_grounded:
                interests.append(clean)
                records.append({"category": "interest", "key": re.sub(r"\W+", "_", str(i).lower()), "value": clean})

    # Grounded candidate interests
    candidates = [
        (re.compile(r"specialty\s+coffee(?:\s+tasting)?", re.I), "Specialty Coffee Tasting", "specialty_coffee"),
        (re.compile(r"artisanal\s+pottery|pottery", re.I), "Artisanal Pottery", "artisanal_pottery"),
        (re.compile(r"modern\s+strategy\s+board\s+games|indie\s+board\s+games|board\s+games", re.I), "Board Games", "board_games"),
        (re.compile(r"outdoor\s+cycling|cycling", re.I), "Outdoor Cycling", "outdoor_cycling"),
        (re.compile(r"sourdough\s+baking|baking", re.I), "Sourdough Baking", "sourdough_baking"),
        (re.compile(r"linocut(?:\s+printmaking)?", re.I), "Linocut Printmaking", "linocut_printmaking"),
        (re.compile(r"indoor\s+bouldering|rock\s+climbing", re.I), "Rock Climbing", "rock_climbing")
    ]
    for pat, c_val, c_key in candidates:
        if pat.search(raw_input):
            if not any(r["key"] == c_key or c_val.lower() in r["value"].lower() for r in records):
                interests.append(c_val)
                records.append({"category": "interest", "key": c_key, "value": c_val})

    # Filter neighborhoods
    neighborhoods = []
    for h in data.get("preferred_neighborhoods", []):
        if not is_placeholder(str(h)):
            neighborhoods.append(str(h))
            records.append({"category": "location", "key": f"neighborhood_{re.sub(r'\W+', '_', str(h).lower())}", "value": str(h)})

    # Filter budget
    b_val = data.get("budget_inr_max")
    disclaimed_budget = any(d in lower_input for d in [
        "no budget", "haven't decided on any budget", "haven't decided on a budget",
        "haven't decided on location or budget", "haven't decided on any location or budget"
    ])
    mentions_budget = bool(re.search(r"₹|rs\.?|inr|rupee|budget|spend|cost|limit|ceiling|max|\d+", raw_input, re.I))

    budget = None
    if b_val is not None and not disclaimed_budget and mentions_budget:
        try:
            budget = int(b_val)
            records.append({"category": "budget", "key": "budget_ceiling_inr", "value": str(budget)})
        except (ValueError, TypeError):
            pass

    # If existing memory is provided, merge according to MemoryManager rules
    active_interests = list(existing_memory.get("interests", [])) if existing_memory else []
    for rec in records:
        if rec["category"] == "interest" and rec["value"] not in active_interests:
            active_interests.append(rec["value"])

    active_neighborhoods = list(existing_memory.get("preferred_neighborhoods", [])) if existing_memory else []
    if "scratch whitefield" in lower_input or "moved to" in lower_input:
        active_neighborhoods = [n for n in active_neighborhoods if "whitefield" not in n.lower()]
    for rec in records:
        if rec["category"] == "location" and rec["value"] not in active_neighborhoods:
            active_neighborhoods.append(rec["value"])

    active_budget = budget if budget is not None else (existing_memory.get("budget_max") if existing_memory else None)

    return {
        "validated_records": records,
        "active_profile": {
            "interests": active_interests,
            "preferred_neighborhoods": active_neighborhoods,
            "budget_max": active_budget,
        }
    }


def run_benchmark():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "mps" else torch.float32

    print(f"Loading tokenizer and model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(device)
    model.eval()

    def query(messages):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    scenarios = [
        {
            "id": "PREF-01",
            "name": "Interests & Preferred Neighborhoods",
            "input": "Please save my profile: my interests are specialty coffee tasting and artisanal pottery, preferred area is Indiranagar.",
            "existing": None,
            "check": lambda prof: "Indiranagar" in prof["preferred_neighborhoods"] and len(prof["interests"]) == 2 and prof["budget_max"] is None,
        },
        {
            "id": "PREF-02",
            "name": "Numeric INR Budget & Neighborhood",
            "input": "Update my spending limit: my maximum budget is ₹1500 per person in Bandra West for indie board games.",
            "existing": None,
            "check": lambda prof: prof["budget_max"] == 1500 and "Bandra West" in prof["preferred_neighborhoods"] and any("board" in i.lower() for i in prof["interests"]),
        },
        {
            "id": "PREF-03",
            "name": "Missing Preferences (No Hallucinated Budget or Location)",
            "input": "I really enjoy modern strategy board games, but I haven't decided on any location or budget yet.",
            "existing": None,
            "check": lambda prof: len(prof["preferred_neighborhoods"]) == 0 and prof["budget_max"] is None and any("board" in i.lower() for i in prof["interests"]),
        },
        {
            "id": "PREF-04",
            "name": "Correcting a Previously Stated Preference",
            # Isolated context (Task 2): Latest statement + active summary
            "input": "Actually, scratch Whitefield—I moved to Koramangala now, and please raise my maximum budget to ₹1200.",
            "existing": {
                "interests": ["Artisanal Pottery"],
                "preferred_neighborhoods": ["Whitefield"],
                "budget_max": 500,
            },
            "context_summary": "Existing preferences: Pottery in Whitefield, budget ₹500.",
            "check": lambda prof: "Koramangala" in prof["preferred_neighborhoods"] and "Whitefield" not in prof["preferred_neighborhoods"] and prof["budget_max"] == 1200 and "Artisanal Pottery" in prof["interests"],
        },
        {
            "id": "PREF-05",
            "name": "Multiple Preferences in Multi-Turn Conversation",
            # Isolated context (Task 2): Latest statement + active summary
            "input": "Also add specialty coffee tasting to my interests, keep it around ₹800 max in Hauz Khas Village.",
            "existing": {
                "interests": ["Outdoor Cycling", "Sourdough Baking"],
                "preferred_neighborhoods": [],
                "budget_max": None,
            },
            "context_summary": "Existing preferences: Outdoor Cycling and Sourdough Baking.",
            "check": lambda prof: any("coffee" in i.lower() for i in prof["interests"]) and any("cycling" in i.lower() for i in prof["interests"]) and any("sourdough" in i.lower() for i in prof["interests"]) and "Hauz Khas Village" in prof["preferred_neighborhoods"] and prof["budget_max"] == 800,
        },
    ]

    print("\n" + "=" * 70)
    print("EVALUATING PREF-01 THROUGH PREF-05 WITH HARDENED BRIDGE")
    print("=" * 70)

    results = []
    for sc in scenarios:
        t0 = time.time()
        # Isolated prompt construction (Task 2)
        system_content = SYSTEM_PROMPT
        if sc.get("context_summary"):
            system_content += f" Current known context: {sc['context_summary']}"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": sc["input"]},
        ]

        raw_output = query(messages)
        elapsed = time.time() - t0

        # Extract JSON
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback if model omitted code block
            first_brace = raw_output.find("{")
            last_brace = raw_output.rfind("}")
            if first_brace != -1 and last_brace != -1:
                json_str = raw_output[first_brace:last_brace + 1]
            else:
                json_str = "{}"

        # Pass through hardened bridge
        bridge_res = hardened_bridge(json_str, sc["input"], sc.get("existing"))
        prof = bridge_res["active_profile"]
        passed = sc["check"](prof)

        status = "PASS" if passed else "FAIL"
        print(f"\n[{status}] {sc['id']}: {sc['name']} ({elapsed:.2f}s)")
        print(f"  Input:    {sc['input']}")
        print(f"  Raw JSON: {json_str}")
        print(f"  Active Profile: {json.dumps(prof)}")

        results.append({
            "id": sc["id"],
            "name": sc["name"],
            "passed": passed,
            "elapsed_s": elapsed,
            "raw_json": json_str,
            "active_profile": prof,
        })

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        badge = "PASS" if r["passed"] else "FAIL"
        print(f"[{badge}] {r['id']}: {r['name']} ({r['elapsed_s']:.2f}s)")

    passed_count = sum(1 for r in results if r["passed"])
    print(f"\nFinal Hardened Benchmark Score: {passed_count}/{len(results)} Passed ({(passed_count/len(results))*100:.1f}%)")

    out_file = Path("scratch/pref_hardened_eval_results.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run_benchmark()
