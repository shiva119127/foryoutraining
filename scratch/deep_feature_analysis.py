#!/usr/bin/env python3
"""
Deep Linguistic & Preference Feature Extraction
"""
import json
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data")

with open(DATA_DIR / "train_expanded_v2.jsonl", "r", encoding="utf-8") as f:
    train_records = [json.loads(line) for line in f if line.strip()]

with open(DATA_DIR / "eval.jsonl", "r", encoding="utf-8") as f:
    eval_records = [json.loads(line) for line in f if line.strip()]

with open(DATA_DIR / "eval_expanded.jsonl", "r", encoding="utf-8") as f:
    eval_exp_records = [json.loads(line) for line in f if line.strip()]

# 1. Activities in train_expanded_v2
activity_counts = Counter()
for rec in train_records:
    user_text = " ".join(m.get("content", "") for m in rec.get("messages", []) if m.get("role") == "user").lower()
    # Check activity keywords
    keywords = [
        "pottery", "ceramic", "acrylic", "candle", "perfume", "linocut", "terrazzo",
        "mosaic", "stained glass", "leather", "terrarium", "sketching", "coffee",
        "sourdough", "matcha", "chocolate", "pasta", "dim sum", "mixology", "kombucha",
        "pizza", "tea", "board game", "card game", "rpg", "trivia", "murder mystery",
        "escape room", "chess", "arcade", "book reading", "sound bath", "nature walk",
        "yoga", "birdwatching", "vinyl", "botanical garden", "watercolor", "journaling",
        "rock climbing", "cycling", "frisbee", "badminton", "slacklining", "kayaking",
        "jog", "table tennis"
    ]
    for kw in keywords:
        if kw in user_text:
            activity_counts[kw] += 1

print("--- ACTIVITY DISTRIBUTION (Top 25) ---")
for act, count in activity_counts.most_common(25):
    print(f"  {act:25s}: {count:5d} ({count/len(train_records)*100:5.2f}%)")

# 2. Language / Code-mixing / Indian English idioms
hinglish_counts = Counter()
hinglish_words = [
    "chai", "adda", "jugaad", "yaar", "bhai", "namaste", "desi", "bhelpuri", "chaat",
    "dosa", "biryani", "auto", "metro", "monsoon", "diwali", "puja", "bazaar", "thali"
]
for rec in train_records:
    user_text = " ".join(m.get("content", "") for m in rec.get("messages", []) if m.get("role") == "user").lower()
    for hw in hinglish_words:
        if re.search(r'\b' + hw + r'\b', user_text):
            hinglish_counts[hw] += 1

print("\n--- HINGLISH / INDIAN VOCABULARY ---")
for hw, count in hinglish_counts.most_common():
    print(f"  {hw:15s}: {count:5d}")
total_hinglish = sum(1 for rec in train_records if any(re.search(r'\b' + hw + r'\b', " ".join(m.get("content", "") for m in rec.get("messages", []) if m.get("role") == "user").lower()) for hw in hinglish_words))
print(f"Total records with any Hinglish/Indian cultural loanwords: {total_hinglish} ({total_hinglish/len(train_records)*100:.2f}%)")

# 3. Templating & Structural Archetypes
template_starts = Counter()
for rec in train_records:
    user_text = " ".join(m.get("content", "") for m in rec.get("messages", []) if m.get("role") == "user").strip()
    # First 4 words
    words = user_text.split()[:4]
    template_starts[" ".join(words)] += 1

print("\n--- TOP 15 USER PROMPT 4-GRAM OPENERS ---")
for opener, count in template_starts.most_common(15):
    print(f"  {opener:40s}: {count:5d}")

# 4. Multi-turn vs Single-turn
turn_counts = Counter(len([m for m in rec["messages"] if m["role"] == "user"]) for rec in train_records)
print("\n--- USER TURNS PER CONVERSATION ---")
for turns, count in turn_counts.items():
    print(f"  {turns} user turn(s): {count:5d} ({count/len(train_records)*100:.2f}%)")

# 5. Extraction vs Conversation breakdown
print("\n--- SYSTEM PERSONAS / ASSISTANT MODES ---")
system_personas = Counter()
for rec in train_records:
    for m in rec.get("messages", []):
        if m.get("role") == "system":
            # first 40 chars
            system_personas[m.get("content", "")[:50]] += 1
for p, c in system_personas.most_common():
    print(f"  '{p}...': {c}")
