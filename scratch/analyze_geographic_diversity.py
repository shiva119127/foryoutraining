#!/usr/bin/env python3
"""
FlowersForYou - Geographic and Preference Diversity Analyzer
Audits:
- data/train_expanded_v2.jsonl (10,000 examples)
- data/eval.jsonl (9 examples)
- data/eval_expanded.jsonl (56 examples)
- data/train.jsonl (37 examples)
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Geographic definitions
CITY_KEYWORDS = {
    "Bengaluru": [
        "bengaluru", "bangalore", "indiranagar", "koramangala", "hsr layout", "hsr",
        "jayanagar", "malleshwaram", "malleswaram", "whitefield", "jp nagar",
        "church street", "sadashivanagar", "bellandur", "basavanagudi", "frazer town",
        "fraser town", "richmond town", "electronic city", "rajajinagar", "sankey tank",
        "ulsoor", "kalyan nagar", "btm layout", "btm", "cubbon park", "lalbagh",
        "mg road", "brigade road", "marathahalli", "sarjapur", "yelahanka", "banashankari",
        "commercial street", "shivaji nagar", "vasanth nagar", "domlur", "cooke town"
    ],
    "Mumbai": [
        "mumbai", "bombay", "bandra west", "bandra east", "bandra", "colaba", "lower parel",
        "juhu", "kala ghoda", "versova", "dadar", "powai", "lokhandwala", "andheri",
        "fort", "matunga", "santacruz", "chembur", "marine drive", "worli", "malabar hill",
        "ghatkopar", "borivali", "vile parle", "khar", "bkc", "nariman point"
    ],
    "Delhi-NCR": [
        "delhi", "new delhi", "delhi-ncr", "gurgaon", "gurugram", "noida", "hauz khas",
        "connaught place", "cyberhub", "cyber hub", "saket", "khan market", "greater kailash",
        "gk", "lodhi", "vasant kunj", "chandni chowk", "south extension", "dwarka",
        "karol bagh", "lajpat nagar"
    ],
    "Pune": [
        "pune", "koregaon park", "baner", "kalyani nagar", "fc road", "viman nagar",
        "aundh", "kothrud", "magarpatta", "wakad", "hinjewadi"
    ],
    "Hyderabad": [
        "hyderabad", "jubilee hills", "banjara hills", "gachibowli", "madhapur",
        "kondapur", "hitec city", "begumpet", "secunderabad", "charminar", "kukatpally"
    ],
    "Chennai": [
        "chennai", "madras", "adyar", "besant nagar", "t nagar", "mylapore",
        "anna nagar", "nungambakkam", "alwarpet", "ecr", "velachery", "omr"
    ],
    "Kolkata": [
        "kolkata", "calcutta", "park street", "salt lake", "new town", "ballygunge",
        "alipore", "gariahat", "howrah"
    ],
    "Other Indian Cities": [
        "ahmedabad", "jaipur", "lucknow", "kochi", "cochin", "chandigarh",
        "coimbatore", "indore", "bhopal", "surat", "vadodara", "nagpur",
        "visakhapatnam", "vizag", "patna", "bhubaneswar", "guwahati", "dehradun",
        "goa", "mysore", "mysuru", "mangalore", "mangaluru", "amritsar",
        "varanasi", "agra", "kanpur", "trivandrum", "thiruvananthapuram"
    ]
}

def detect_locations(text: str) -> dict:
    text_lower = text.lower()
    matches = defaultdict(list)
    for city, kws in CITY_KEYWORDS.items():
        for kw in kws:
            # Word boundary search
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, text_lower):
                matches[city].append(kw)
    return matches

def analyze_file(filepath: Path):
    print(f"\n========================================================")
    print(f"AUDITING FILE: {filepath.name} ({filepath})")
    print(f"========================================================")
    
    with open(filepath, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    total_records = len(records)
    print(f"Total Records: {total_records}")

    # Location tracking
    user_locations = Counter()
    assistant_locations = Counter()
    user_city_counts = Counter()
    assistant_city_counts = Counter()
    
    # Specific location classifications
    records_with_user_loc = 0
    records_with_output_only_loc = 0
    records_with_no_loc = 0
    records_multi_city = 0

    # Specific Bengaluru breakdowns
    bengaluru_neighborhoods = Counter()

    # Preference domains
    budgets = []
    has_budget = 0
    budget_values = Counter()
    clear_budget_count = 0

    dietary_counts = Counter()
    accessibility_counts = Counter()
    intent_categories = Counter()
    
    # Activity tracking
    activity_words = Counter()

    # User texts for near-duplicate analysis
    user_prompts = []
    
    for idx, rec in enumerate(records):
        messages = rec.get("messages", [])
        user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
        assistant_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "assistant")
        sys_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "system")

        user_prompts.append(user_text)

        # Detect category from system message or metadata
        cat = "unknown"
        if "category" in rec:
            cat = rec["category"]
        else:
            if "onboarding assistant" in sys_text:
                cat = "onboarding"
            elif "preference extractor" in sys_text:
                cat = "extraction"
            elif "planner" in sys_text:
                cat = "planner"
            elif "returning user" in sys_text or "returning" in sys_text:
                cat = "returning"
        intent_categories[cat] += 1

        # Detect locations in User Input
        user_loc_map = detect_locations(user_text)
        # Detect locations in Assistant Input
        asst_loc_map = detect_locations(assistant_text)

        if user_loc_map:
            records_with_user_loc += 1
            if len(user_loc_map.keys()) > 1:
                records_multi_city += 1
            for city, kws in user_loc_map.items():
                user_city_counts[city] += 1
                for kw in kws:
                    user_locations[kw] += 1
                    if city == "Bengaluru":
                        bengaluru_neighborhoods[kw] += 1
        elif asst_loc_map:
            records_with_output_only_loc += 1
            for city, kws in asst_loc_map.items():
                assistant_city_counts[city] += 1
                for kw in kws:
                    assistant_locations[kw] += 1
        else:
            records_with_no_loc += 1

        # Budget extraction
        b_match = re.findall(r'(?:₹|rs\.?|inr)\s*(\d+)', user_text.lower()) + re.findall(r'(\d+)\s*(?:rupees|inr|per person|total)', user_text.lower())
        if b_match:
            has_budget += 1
            for b in b_match:
                try:
                    val = int(b)
                    if 50 <= val <= 50000:
                        budget_values[val] += 1
                        budgets.append(val)
                except ValueError:
                    pass
        if "clear" in user_text.lower() and "budget" in user_text.lower() or "no budget" in user_text.lower() or "don't want any budget" in user_text.lower():
            clear_budget_count += 1

        # Dietary
        for d in ["vegan", "vegetarian", "jain", "gluten-free", "dairy-free", "halal", "eggless", "nut allergy"]:
            if d in user_text.lower():
                dietary_counts[d] += 1

        # Accessibility
        for a in ["wheelchair", "step-free", "elevator", "ramp", "accessible", "quiet space", "sensory", "guide dog"]:
            if a in user_text.lower():
                accessibility_counts[a] += 1

    print("\n--- 1. GEOGRAPHIC CONCENTRATION (USER INPUTS) ---")
    print(f"Records with explicit user location: {records_with_user_loc} ({records_with_user_loc/total_records*100:.2f}%)")
    print(f"Records with output-only location:    {records_with_output_only_loc} ({records_with_output_only_loc/total_records*100:.2f}%)")
    print(f"Records with NO location (Generic):   {records_with_no_loc} ({records_with_no_loc/total_records*100:.2f}%)")
    print(f"Records with multi-city mentions:     {records_multi_city} ({records_multi_city/total_records*100:.2f}%)")

    print("\nUser-Mentioned City Breakdown (denominator = total records):")
    for city, count in user_city_counts.most_common():
        pct = (count / total_records) * 100
        pct_of_loc = (count / records_with_user_loc * 100) if records_with_user_loc else 0
        print(f"  - {city:20s}: {count:5d} ({pct:5.2f}% of all records, {pct_of_loc:5.2f}% of located)")

    if "Bengaluru" in user_city_counts:
        print("\nTop Bengaluru Neighborhoods Mentioned in User Input:")
        for neigh, count in bengaluru_neighborhoods.most_common(15):
            print(f"    * {neigh:25s}: {count:5d}")

    print("\n--- 2. PREFERENCE DIVERSITY (BUDGETS, DIETS, ACCESS) ---")
    print(f"Records with explicit budget numbers: {has_budget} ({has_budget/total_records*100:.2f}%)")
    print(f"Records clearing/removing budget:     {clear_budget_count} ({clear_budget_count/total_records*100:.2f}%)")
    if budgets:
        budgets.sort()
        print(f"Budget min: ₹{min(budgets)}, med: ₹{budgets[len(budgets)//2]}, max: ₹{max(budgets)}")
        print("Top 10 budget values:")
        for b, c in budget_values.most_common(10):
            print(f"    * ₹{b}: {c} occurrences")

    print("\nDietary Preferences in User Input:")
    for d, c in dietary_counts.most_common():
        print(f"  - {d:20s}: {c:5d} ({c/total_records*100:.2f}%)")

    print("\nAccessibility Requirements in User Input:")
    for a, c in accessibility_counts.most_common():
        print(f"  - {a:20s}: {c:5d} ({c/total_records*100:.2f}%)")

    print("\n--- 3. USER INTENT CATEGORIES ---")
    for cat, count in intent_categories.most_common():
        print(f"  - {cat:30s}: {count:5d} ({count/total_records*100:.2f}%)")

    # Uniqueness check on user inputs
    unique_prompts = len(set(" ".join(p.lower().split()) for p in user_prompts))
    print(f"\nUnique user prompts (exact normalized string): {unique_prompts} / {total_records} ({unique_prompts/total_records*100:.2f}%)")

    return {
        "total": total_records,
        "records_with_user_loc": records_with_user_loc,
        "records_with_no_loc": records_with_no_loc,
        "user_city_counts": dict(user_city_counts),
        "bengaluru_neighborhoods": dict(bengaluru_neighborhoods),
        "has_budget": has_budget,
        "dietary_counts": dict(dietary_counts),
        "accessibility_counts": dict(accessibility_counts),
        "unique_prompts": unique_prompts,
        "user_prompts": user_prompts
    }

if __name__ == "__main__":
    t_v2 = analyze_file(DATA_DIR / "train_expanded_v2.jsonl")
    e_std = analyze_file(DATA_DIR / "eval.jsonl")
    e_exp = analyze_file(DATA_DIR / "eval_expanded.jsonl")
    t_std = analyze_file(DATA_DIR / "train.jsonl")

    # Cross-dataset leakage check
    print("\n========================================================")
    print("CROSS-DATASET OVERLAP & LEAKAGE AUDIT")
    print("========================================================")
    eval_user_prompts = set(" ".join(p.lower().split()) for p in e_std["user_prompts"])
    eval_exp_user_prompts = set(" ".join(p.lower().split()) for p in e_exp["user_prompts"])
    train_prompts = set(" ".join(p.lower().split()) for p in t_v2["user_prompts"])

    exact_leak_eval = eval_user_prompts & train_prompts
    exact_leak_eval_exp = eval_exp_user_prompts & train_prompts

    print(f"Exact string matches between train_v2 and eval.jsonl: {len(exact_leak_eval)}")
    print(f"Exact string matches between train_v2 and eval_expanded.jsonl: {len(exact_leak_eval_exp)}")
