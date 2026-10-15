#!/usr/bin/env python3
"""
FlowersForYou - Repaired Dataset Generator (v2).
Phase 6C: High-Diversity, Grammatically Sound, Zero-Leakage Dataset Expansion.

Fixes Phase 6B quality findings:
1. Eliminates near-duplicate stems (strictly Jaccard < 0.80 globally across all records).
2. Fixes grammatical phrasing for plural/gerund activities (clean determiners).
3. Fixes skip-onboarding context amnesia (acknowledges pre-stated locations).
4. Radically expands diversity across all 11 behavioral domains with dynamic combinatorial clause composition.
5. Strict zero-leakage against data/eval.jsonl and data/eval_expanded.jsonl.
6. Outputs data/train_expanded_v2.jsonl and data/dataset_manifest_v2.jsonl without touching v1 artifacts.
"""

import argparse
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluate import evaluate_response

DATA_DIR = ROOT / "data"
BATCHES_DIR = DATA_DIR / "generated_batches_v2"
REPORTS_DIR = ROOT / "reports"
BATCH_REPORTS_DIR = REPORTS_DIR / "dataset_batches_v2"
EVAL_FILE = DATA_DIR / "eval.jsonl"
EVAL_EXPANDED_FILE = DATA_DIR / "eval_expanded.jsonl"

EXPANDED_TRAIN_V2 = DATA_DIR / "train_expanded_v2.jsonl"
MANIFEST_V2 = DATA_DIR / "dataset_manifest_v2.jsonl"

CATEGORY_BATCH_QUOTAS = {
    "first_time_onboarding": 40,
    "activity_discovery": 65,
    "structured_preference_extraction": 35,
    "returning_users": 30,
    "constraints_handling": 45,
    "multi_turn_corrections": 60,
    "undecided_and_skip_onboarding": 35,
    "tool_grounded_search": 50,
    "booking_safety_and_verification": 70,
    "ambiguous_and_privacy": 35,
    "feedback_and_recovery": 35,
}

# -------------------------------------------------------------
# Rich Indian Urban Locality Contexts
# -------------------------------------------------------------
CITIES_AREAS = {
    "Bengaluru": [
        "Indiranagar", "Koramangala", "HSR Layout", "Jayanagar", "Malleshwaram",
        "Whitefield", "JP Nagar", "Church Street", "Sadashivanagar", "Bellandur",
        "Basavanagudi", "Frazer Town", "Richmond Town", "Electronic City", "Rajajinagar",
        "Sankey Tank area", "Ulsoor Lake area", "Kalyan Nagar", "BTM Layout"
    ],
    "Mumbai": [
        "Bandra West", "Colaba", "Lower Parel", "Juhu", "Kala Ghoda", "Versova",
        "Dadar", "Powai", "Andheri Lokhandwala", "Fort", "Matunga", "Santacruz", "Chembur"
    ],
    "Delhi-NCR": [
        "Hauz Khas Village", "Connaught Place", "CyberHub Gurgaon", "Saket",
        "Khan Market", "Greater Kailash", "Lodhi Art District", "Vasant Kunj"
    ],
    "Pune": [
        "Koregaon Park", "Baner", "Kalyani Nagar", "FC Road", "Viman Nagar", "Aundh"
    ],
    "Hyderabad": [
        "Jubilee Hills", "Banjara Hills", "Gachibowli", "Madhapur", "Kondapur"
    ]
}

# -------------------------------------------------------------
# Grammatically Categorized Activities
# -------------------------------------------------------------
ACTIVITIES = {
    "creative": [
        {"slug": "wheel_pottery_workshop", "name": "wheel pottery workshop", "with_art": "a wheel pottery workshop", "action": "try a wheel pottery workshop"},
        {"slug": "ceramic_glaze_painting", "name": "ceramic glaze painting", "with_art": "a ceramic glaze painting session", "action": "try ceramic glaze painting"},
        {"slug": "fluid_acrylic_canvas_art", "name": "fluid acrylic canvas art", "with_art": "a fluid acrylic canvas art session", "action": "explore fluid acrylic canvas art"},
        {"slug": "soy_candle_making", "name": "soy candle making", "with_art": "a soy candle making workshop", "action": "learn soy candle making"},
        {"slug": "perfume_blending_session", "name": "perfume blending session", "with_art": "a perfume blending workshop", "action": "attend a perfume blending session"},
        {"slug": "linocut_printmaking", "name": "linocut printmaking", "with_art": "a linocut printmaking workshop", "action": "learn linocut printmaking"},
        {"slug": "terrazzo_tray_crafting", "name": "terrazzo tray crafting", "with_art": "a terrazzo tray crafting session", "action": "try terrazzo tray crafting"},
        {"slug": "mosaic_tile_crafting", "name": "mosaic tile crafting", "with_art": "a mosaic tile crafting workshop", "action": "craft with mosaic tiles"},
        {"slug": "stained_glass_crafting", "name": "stained glass crafting", "with_art": "a stained glass workshop", "action": "try stained glass crafting"},
        {"slug": "leather_craft_basics", "name": "leather craft basics", "with_art": "a leather craft basics workshop", "action": "learn leather craft basics at a workshop"},
        {"slug": "miniature_terrarium_building", "name": "miniature terrarium building", "with_art": "a miniature terrarium workshop", "action": "build a miniature terrarium"},
        {"slug": "botanical_sketching", "name": "botanical sketching", "with_art": "a botanical sketching walk", "action": "join a botanical sketching walk"}
    ],
    "culinary": [
        {"slug": "specialty_coffee_tasting", "name": "specialty coffee tasting", "with_art": "a specialty coffee tasting session", "action": "attend a specialty coffee tasting session"},
        {"slug": "artisanal_sourdough_baking", "name": "artisanal sourdough baking", "with_art": "an artisanal sourdough baking masterclass", "action": "learn artisanal sourdough baking"},
        {"slug": "matcha_tea_ceremony", "name": "matcha tea ceremony", "with_art": "a matcha tea ceremony session", "action": "experience a matcha tea ceremony"},
        {"slug": "bean_to_bar_chocolate_making", "name": "bean-to-bar chocolate making", "with_art": "a bean-to-bar chocolate making workshop", "action": "make bean-to-bar chocolate"},
        {"slug": "pasta_making_from_scratch", "name": "pasta making from scratch", "with_art": "a fresh pasta making masterclass", "action": "take a fresh pasta making masterclass"},
        {"slug": "dim_sum_masterclass", "name": "dim sum masterclass", "with_art": "a dim sum masterclass", "action": "take a dim sum masterclass"},
        {"slug": "pour_over_coffee_brewing", "name": "pour-over coffee brewing", "with_art": "a pour-over coffee brewing class", "action": "learn pour-over coffee brewing"},
        {"slug": "cheese_and_fruit_pairing", "name": "cheese and fruit pairing", "with_art": "an artisanal cheese pairing session", "action": "explore cheese and fruit pairing"},
        {"slug": "mocktail_mixology_class", "name": "mocktail mixology class", "with_art": "a craft mocktail mixology class", "action": "learn mocktail mixology"},
        {"slug": "fermentation_and_kombucha_workshop", "name": "fermentation and kombucha workshop", "with_art": "a fermentation and kombucha workshop", "action": "learn kombucha brewing"},
        {"slug": "artisan_pizza_baking", "name": "artisan pizza baking", "with_art": "a wood-fired pizza masterclass", "action": "bake artisan pizzas"},
        {"slug": "tea_blending_class", "name": "tea blending class", "with_art": "a custom tea blending class", "action": "blend custom specialty teas"}
    ],
    "social_games": [
        {"slug": "modern_strategy_board_games", "name": "modern strategy board games", "with_art": "a modern strategy board game session", "action": "play modern strategy board games"},
        {"slug": "indie_card_games", "name": "indie card games", "with_art": "an indie card game meetup", "action": "play indie card games"},
        {"slug": "tabletop_rpg_one_shot", "name": "tabletop RPG one-shot", "with_art": "a tabletop RPG one-shot campaign", "action": "play a tabletop RPG one-shot"},
        {"slug": "casual_pub_trivia_quiz", "name": "casual pub trivia quiz", "with_art": "a casual pub trivia quiz evening", "action": "compete in a pub trivia quiz"},
        {"slug": "murder_mystery_deduction_game", "name": "murder mystery deduction game", "with_art": "a murder mystery deduction game evening", "action": "play a murder mystery deduction game"},
        {"slug": "cooperative_escape_room", "name": "cooperative escape room", "with_art": "a cooperative escape room challenge", "action": "solve a cooperative escape room"},
        {"slug": "chess_meetup_in_the_park", "name": "chess meetup in the park", "with_art": "an outdoor chess meetup", "action": "play in an outdoor chess meetup"},
        {"slug": "social_word_game_evening", "name": "social word-game evening", "with_art": "a social word-game evening", "action": "join a social word-game evening"},
        {"slug": "vintage_arcade_games", "name": "vintage arcade games", "with_art": "a vintage arcade gaming session", "action": "play vintage arcade games"}
    ],
    "relaxed_wellness": [
        {"slug": "quiet_book_reading_club", "name": "quiet book reading club", "with_art": "a quiet book reading circle", "action": "join a quiet book reading club"},
        {"slug": "rooftop_sunset_sound_bath", "name": "rooftop sunset sound bath", "with_art": "a sunset sound bath session", "action": "experience a sunset sound bath"},
        {"slug": "guided_morning_nature_walk", "name": "guided morning nature walk", "with_art": "a guided morning nature walk", "action": "join a guided morning nature walk"},
        {"slug": "mindful_outdoor_yoga", "name": "mindful outdoor yoga", "with_art": "a mindful outdoor yoga class", "action": "practice in a mindful outdoor yoga class"},
        {"slug": "birdwatching_at_the_lake", "name": "birdwatching at the lake", "with_art": "a morning birdwatching walk", "action": "go birdwatching at the lake"},
        {"slug": "ambient_vinyl_listening_lounge", "name": "ambient vinyl listening lounge", "with_art": "an ambient vinyl listening session", "action": "relax in an ambient vinyl listening lounge"},
        {"slug": "botanical_garden_stroll", "name": "botanical garden stroll", "with_art": "a relaxing botanical garden stroll", "action": "take a botanical garden stroll"},
        {"slug": "watercolor_painting_in_the_park", "name": "watercolor painting in the park", "with_art": "an outdoor watercolor painting meetup", "action": "paint watercolors in the park"},
        {"slug": "meditative_journaling", "name": "meditative journaling", "with_art": "a guided meditative journaling workshop", "action": "practice in a guided meditative journaling workshop"}
    ],
    "active_outdoor": [
        {"slug": "indoor_rock_climbing", "name": "indoor rock climbing", "with_art": "an indoor rock climbing session", "action": "try indoor rock climbing"},
        {"slug": "evening_heritage_cycling_tour", "name": "evening heritage cycling tour", "with_art": "an evening heritage cycling tour", "action": "join an evening heritage cycling tour"},
        {"slug": "outdoor_lawn_frisbee_meetup", "name": "outdoor lawn frisbee meetup", "with_art": "a casual lawn frisbee meetup", "action": "play in a lawn frisbee meetup"},
        {"slug": "badminton_doubles_session", "name": "badminton doubles session", "with_art": "a badminton doubles session", "action": "play a badminton doubles match"},
        {"slug": "slacklining_in_the_park", "name": "slacklining in the park", "with_art": "a park slacklining workshop", "action": "try slacklining in the park"},
        {"slug": "kayaking_on_the_lake", "name": "kayaking on the lake", "with_art": "a lake kayaking session", "action": "go kayaking on the lake"},
        {"slug": "morning_trail_jog", "name": "morning trail jog", "with_art": "a morning trail group jog", "action": "join a morning trail jog"},
        {"slug": "table_tennis_meetup", "name": "table tennis meetup", "with_art": "a friendly table tennis meetup", "action": "play in a table tennis meetup"}
    ]
}

ALL_ACTIVITIES = [item for cat_list in ACTIVITIES.values() for item in cat_list]

VENUES_CATALOG = [
    "Blue Tokai Coffee Roasters", "Third Wave Coffee", "Dabbler's Den", "Clay Station",
    "Dice & Dine Cafe", "Chai Point", "Subko Specialty Coffee", "Araku Coffee",
    "The Board Room", "Artistic Soul Studio", "Pottery Town Workshop", "Cafe Noir",
    "Bakehouse 7", "Doolally Taproom", "Dialogues Cafe", "Atta Galatta", "Champaca Bookstore",
    "The Crafter's House", "The Pottery Lab", "Toit Brewpub", "Brik Oven", "Roastery Coffee House",
    "True Black Coffee", "Kala Ghoda Cafe", "Subko Bandra", "Greenr Cafe", "Blue Door Cafe",
    "Sly Granny", "Fabcafe by the Lake", "The Grammar Room", "Zen Cafe", "Pagdandi Books Chai",
    "Leaping Windows", "Chaayos", "The Table", "Nature's Basket Artisanal Studio", "Smash Tennis Academy",
    "Climb Central", "Urban Bouldering Hub", "Studio Vasant", "Hauz Khas Pottery Guild", "Smoke House Deli",
    "The Bombay Canteen", "SodaBottleOpenerWala", "Paul Bakery Cafe", "The Bagel Shop", "Candies Bandra",
    "Madras Glass Guild", "Katha Cafe", "Hoppipola", "Under the Mango Tree", "The Clay Company"
]


def tokenize(text: str) -> set:
    return set(re.findall(r"\w+", text.lower()))


def jaccard_similarity(s1: str, s2: str) -> float:
    t1 = tokenize(s1)
    t2 = tokenize(s2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def get_conversation_user_text(messages: list) -> str:
    return " ".join(m.get("content", "") for m in messages if m.get("role") == "user")


class DiversityIndex:
    """Fast token-based inverted index to guarantee near-duplicate rejection across all 10,000 records."""
    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold
        self.by_len = defaultdict(list)
        self.seen_texts = set()

    def is_near_duplicate(self, text: str) -> bool:
        norm = " ".join(text.lower().split())
        if norm in self.seen_texts:
            return True
        toks = tokenize(norm)
        l1 = len(toks)
        if l1 == 0:
            return True

        min_l2 = int(l1 * self.threshold)
        max_l2 = int(l1 / self.threshold)

        for l2 in range(min_l2, max_l2 + 1):
            for existing_idx, existing_toks in self.by_len[l2]:
                inter = len(toks & existing_toks)
                sim = inter / (l1 + l2 - inter)
                if sim >= self.threshold:
                    return True
        return False

    def add(self, text: str, idx: int):
        norm = " ".join(text.lower().split())
        self.seen_texts.add(norm)
        toks = tokenize(norm)
        self.by_len[len(toks)].append((idx, toks))


class RepairedScenarioEngine:
    """Generates linguistically diverse, non-repetitive conversational candidates."""

    def __init__(self, held_out_texts: list[str]):
        self.held_out_texts = held_out_texts

    def is_eval_leak(self, user_text: str) -> bool:
        for ev in self.held_out_texts:
            if jaccard_similarity(user_text, ev) >= 0.38:
                return True
        return False

    # -------------------------------------------------------------
    # 1. First-Time Onboarding (Dynamic Combinatorial Clauses)
    # -------------------------------------------------------------
    def generate_first_time_onboarding(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        area = rng.choice(CITIES_AREAS[city])
        budget = rng.choice([350, 450, 500, 600, 750, 900, 1000, 1200, 1500])

        openers = [
            "Hey there,", "Hello FlowersForYou,", "Good morning,", "Hi team,", "Quick question:",
            "Hey,", "Hi!", "Warm greetings,", "Just reaching out,", "Hello,", "Hey folks,",
            "Hope you are having a good week,", "Starting my weekend planning,", "Weekend is finally here,",
            "Planning ahead for Saturday,", "Looking ahead to Sunday,", "Need some guidance,",
            "First time using FlowersForYou,", "A friend recommended I try this,", "Giving this a spin,",
            "Hoping to get some advice,", "Looking for weekend ideas,", "Decided to try FlowersForYou,",
            "Just downloaded the app,", "Checking this out for the first time,"
        ]

        personas = [
            f"I recently relocated near {area} for a software role",
            f"I moved to an apartment in {area} last month",
            f"I have been living around {area} for a few months now",
            f"I work from home all week in {area} and get bad cabin fever",
            f"My spouse and I recently shifted to {area}",
            f"I am an introvert living near {area} who wants to socialize gently",
            f"I just finished a grueling sprint at my startup in {area}",
            f"I am spending a quiet weekend alone in {area}",
            f"I am an architect new to {area} with an eye for design",
            f"I am a graduate student staying near {area}",
            f"I am an early riser based around {area}",
            f"I have lived in {city} for years but never explored {area} properly",
            f"I am hosting two friends from college in {area}",
            f"I want to break the monotony of binge-watching in {area}",
            f"I am trying to build healthy offline weekend habits around {area}",
            f"I am a freelance designer in {area} needing fresh air",
            f"I am an empty nester in {area} looking for engaging weekend hobbies",
            f"I recently graduated and moved to {area}",
            f"I am new in town, currently living around {area}",
            f"I work long corporate hours in {area} and need a real weekend escape",
            f"I am living solo in {area} and looking for comfortable community spaces",
            f"My spouse and I are looking to explore {area}",
            f"I am an amateur artist staying near {area}",
            f"I have free Saturdays in {area} and want to meet people naturally"
        ]

        motivations = [
            "and want to discover low-key activities to decompress on weekends",
            "and looking for relaxed hobby groups that don't involve loud bars",
            "and really need a reason to step outside and meet people on Saturdays",
            "and want to explore what locals actually do for fun around here",
            "and trying to build a creative weekend routine instead of watching screens",
            "and interested in calm, beginner-friendly workshops where I can learn something new",
            "and looking for peaceful environments over noisy weekend crowds",
            "and we'd love recommendations for daytime cultural or artisanal outings",
            "and trying to find solo activities where it doesn't feel awkward attending alone",
            "and hoping to pick up a hands-on tactile craft like pottery or printmaking",
            "and wanting to explore specialty coffee, sourdough baking, or culinary sessions",
            "and looking for casual social meetups centered on an activity rather than forced small talk",
            "and hoping to find outdoor nature walks, sketching clubs, or board game meetups",
            "and wanting something mindful and unhurried to recharge after a stressful week",
            "and eager to try interactive workshops where introverts can feel comfortable"
        ]

        constraints = [
            "I prefer small intimate groups of under 5 people",
            "Keeping noise levels low is very important to me",
            f"My budget is modest, ideally staying under ₹{budget}",
            "It has to be beginner-friendly with no prior experience required",
            "I need a venue that is easily accessible by public transit or metro",
            "Looking for a morning slot between 10 AM and 1 PM",
            "Prefer an indoor space with air conditioning",
            "Needs to be an environment where attending alone feels completely natural",
            "Hoping for something unhurried that takes around two hours",
            "I want to avoid aggressive sales pitches or corporate networking vibes"
        ]

        closers = [
            "Where would you suggest I begin?", "How do you recommend getting started?",
            "What kind of options fit this best?", "Can you guide me through a couple of possibilities?",
            "Could you point me toward a good starting point?", "What would you recommend for my first weekend outing?",
            "Any pleasant spots or sessions come to mind?", "What are some good starting activities around here?",
            "Where should I look first?", "Would love to hear your recommendations!"
        ]

        parts = []
        if rng.random() < 0.7:
            parts.append(rng.choice(openers))
        parts.append(rng.choice(personas))
        parts.append(rng.choice(motivations))
        if rng.random() < 0.6:
            parts.append(rng.choice(constraints) + ".")
        parts.append(rng.choice(closers))
        u_text = " ".join(parts)

        act_pair = rng.sample(ALL_ACTIVITIES, 2)
        act1 = act_pair[0]
        act2 = act_pair[1]

        assist_responses = [
            f"Welcome to {area}! Settling into a new neighborhood is exciting. To ease into weekend plans, would you enjoy {act1['with_art']}, or something like {act2['with_art']}?",
            f"Welcome! Getting out of the house after a long week is so important. Would you prefer a tactile session like {act1['with_art']}, or a relaxed outing like {act2['with_art']}?",
            f"Glad you reached out! Solo outings are a wonderful way to recharge. To start gently, does {act1['with_art']} sound appealing, or would you lean toward {act2['with_art']}?",
            f"No long questionnaires here! We can take things one step at a time for {area}. Would you be in the mood for {act1['with_art']}, or a fun plan like {act2['with_art']}?",
            f"Small-group and low-pressure meetups make for the best experiences. In {area}, would you lean more toward {act1['with_art']}, or would you prefer {act2['with_art']}?",
            f"Sunday mornings are ideal for calm discovery in {area}. Would you enjoy {act1['with_art']}, or would you prefer trying {act2['with_art']}?",
            f"Activity-based meetups take all the social awkwardness away. For your first outing in {area}, would you prefer {act1['with_art']}, or something like {act2['with_art']}?"
        ]
        a_text = rng.choice(assist_responses)
        sys_prompt = "You are the FlowersForYou onboarding assistant. Learn preferences naturally. Ask one relevant question at a time. Do not turn onboarding into a rigid survey."
        return {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": u_text},
                {"role": "assistant", "content": a_text}
            ]
        }

    # -------------------------------------------------------------
    # 2. Activity Discovery & Planning (Dynamic Combinatorial)
    # -------------------------------------------------------------
    def generate_activity_discovery(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        area = rng.choice(CITIES_AREAS[city])
        cat_key = rng.choice(list(ACTIVITIES.keys()))
        act1, act2 = rng.sample(ACTIVITIES[cat_key], 2)
        budget = rng.choice([350, 450, 500, 600, 750, 850, 950, 1100, 1250, 1500, 1800, 2200])
        timing = rng.choice(["Saturday morning", "Saturday afternoon", "Sunday morning", "Sunday afternoon", "Friday post-work evening", "weekend brunch hours"])

        personas = [
            f"Planning my {timing} around {area}",
            f"Looking ahead to {timing} in {area}",
            f"Wanting to organize something fun for {timing} near {area}",
            f"My friend and I are free this {timing} in {area}",
            f"I have a few hours open this {timing} around {area}",
            f"Putting together weekend plans for {timing} in {area}",
            f"Need inspiration for {timing} near {area}",
            f"Hoping to get out of the house this {timing} around {area}",
            f"Searching for daytime outings for {timing} in {area}"
        ]

        action_phrases = [
            f"in the mood to {act1['action']} or try {cat_key.replace('_', ' ')} activities",
            f"interested in exploring {cat_key.replace('_', ' ')} sessions like {act1['name']}",
            f"curious about {act1['with_art']} or similar {cat_key.replace('_', ' ')} outings",
            f"wanting to try {act1['with_art']} or something hands-on",
            f"hoping to sign up for {act1['with_art']} or related hobbies",
            f"looking for beginner-friendly {cat_key.replace('_', ' ')} options"
        ]

        budget_clauses = [
            f"with a spending limit of ₹{budget}",
            f"keeping total expenses under ₹{budget}",
            f"budget cap is strictly ₹{budget}",
            f"aiming to stay within ₹{budget} per person",
            f"with roughly ₹{budget} to spend",
            f"under a ₹{budget} ceiling"
        ]

        closers = [
            "Can you suggest a couple of verified options?",
            "What fits this best?",
            "Any recommendations that match?",
            "Where should we look first?",
            "Could you share a couple of distinct ideas?",
            "What would you suggest?"
        ]

        u_text = f"{rng.choice(personas)}, {rng.choice(action_phrases)} {rng.choice(budget_clauses)}. {rng.choice(closers)}"

        a_text = (
            f"Here are two great options in {area} within your ₹{budget} budget: you could {act1['action']}, "
            f"or join {act2['with_art']}. Both fit your {timing} window nicely. Are you planning to attend solo or with someone?"
        )
        sys_prompt = "You are the FlowersForYou planner. Onboarding is complete. Gather missing details before searching."
        return {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": u_text},
                {"role": "assistant", "content": a_text}
            ]
        }

    # -------------------------------------------------------------
    # 3. Structured Preference Extraction (Strict Canonical JSON)
    # -------------------------------------------------------------
    def generate_structured_preference_extraction(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        area = rng.choice(CITIES_AREAS[city])
        cat = rng.choice(list(ACTIVITIES.keys()))
        acts = rng.sample(ACTIVITIES[cat], 2)
        slugs = [a["slug"] for a in acts]
        names = [a["name"] for a in acts]
        budget = rng.choice([400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 2500])

        preambles = [
            "Please update my profile preferences:", "Save my preferences:",
            "Here are my profile details to record:", "Can you record my preferences?",
            "Save this to my profile:", "Please store these preferences for me:",
            "Update my account settings with these tastes:", "Record my weekend profile:",
            "Here is my profile info:", "Kindly remember these preferences for upcoming outings:",
            "Store these details in my FlowersForYou profile:", "Updating my saved preferences:"
        ]

        variations = [
            f"{rng.choice(preambles)} I'm really keen on {names[0]} and {names[1]}, my spending cap is ₹{budget}, and I prefer activities in {area}.",
            f"{rng.choice(preambles)} I enjoy {names[0]} along with {names[1]}. Keep my budget around ₹{budget} max in {area}.",
            f"{rng.choice(preambles)} I like {names[0]} and {names[1]}, prefer {area}, and my upper budget limit is ₹{budget}.",
            f"{rng.choice(preambles)} Preferred location is {area}, interests are {names[0]} and {names[1]}, and maximum budget is ₹{budget}.",
            f"{rng.choice(preambles)} For my weekend profile in {area}, budget ceiling is ₹{budget}, and I want to focus on {names[0]} and {names[1]}.",
            f"{rng.choice(preambles)} I'm based around {area}, budget limit ₹{budget}, and my top two interests are {names[0]} and {names[1]}."
        ]
        u_text = rng.choice(variations)

        json_obj = {
            "action": "save_preferences",
            "interests": slugs,
            "preferred_neighborhoods": [area],
            "budget_inr_max": budget
        }
        json_str = json.dumps(json_obj, indent=2)
        a_text = (
            f"```json\n{json_str}\n```\n"
            f"Preferences saved! I've recorded {names[0]}, {names[1]}, a budget ceiling of ₹{budget}, and {area} as your preferred area."
        )
        sys_prompt = "You are the FlowersForYou preference extractor. Extract stated preferences as valid JSON and do not invent details."
        return {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": u_text},
                {"role": "assistant", "content": a_text}
            ]
        }

    # -------------------------------------------------------------
    # 4. Returning Users & Saved Context (No Survey Restart)
    # -------------------------------------------------------------
    def generate_returning_users(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        area = rng.choice(CITIES_AREAS[city])
        act = rng.choice(ALL_ACTIVITIES)

        greetings = [
            "Good to be back!", "Hey FlowersForYou, ready for another weekend plan.",
            "Back again for this week's outing!", "Happy Friday! Looking to line up something fun.",
            "Hello again, ready to plan my Sunday.", "Back after a busy fortnight, ready for a recharge.",
            "Hey team, looking for fresh recommendations for this Saturday.", "Ready for my next plan in town!",
            "Back for more weekend inspiration!", "Hey! Loved my last outing, ready to plan the next one.",
            "Hello again, FlowersForYou! Ready to plan my next adventure.", "Checking in for this weekend's lineup."
        ]

        intentions = [
            f"Sticking around {area} this time.",
            f"Let's focus on spots around {area} for this weekend.",
            f"Looking for something engaging in {area}.",
            f"Hoping to discover a new nook in {area}.",
            f"Looking to explore {area} on Saturday morning.",
            f"Wanting a relaxing session in {area} on Sunday."
        ]
        u_text = f"{rng.choice(greetings)} {rng.choice(intentions)}"

        a_responses = [
            f"Welcome back! Great to plan with you again. Based on your saved profile, we can look into {act['with_art']} in {area}. What time window works best for you this weekend?",
            f"Welcome back! Let's get something wonderful lined up in {area}. Would you like to stick with {act['name']}, or try a completely fresh activity style?",
            f"Always a pleasure to help you plan! We can check verified weekend sessions around {area}. Are you heading out solo this time or catching up with friends?",
            f"Glad to see you back! Let's put together a relaxing plan in {area}. What kind of pace are you in the mood for this Saturday or Sunday?"
        ]
        a_text = rng.choice(a_responses)
        sys_prompt = "You are the FlowersForYou assistant. Onboarding is complete. Do not restart onboarding."
        return {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": u_text},
                {"role": "assistant", "content": a_text}
            ]
        }

    # -------------------------------------------------------------
    # 5. Constraints Handling (Budget, Location, Group, Access)
    # -------------------------------------------------------------
    def generate_constraints_handling(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        area = rng.choice(CITIES_AREAS[city])
        budget = rng.choice([300, 450, 500, 600, 750, 900, 1000, 1200, 1500, 2000, 2500, 3500])
        group = rng.choice([1, 2, 3, 4, 5, 6, 7, 8])

        constraints = [
            ("step-free wheelchair accessibility", "accessibility"),
            ("strictly pure vegetarian or Jain culinary preparations", "dietary"),
            ("a low-noise environment suitable for comfortable conversation", "acoustic"),
            ("verified pet-friendly outdoor seating for a dog", "pet_friendly"),
            ("air-conditioned indoor venue due to peak afternoon heat", "climate"),
            ("dedicated valet parking or close proximity to a metro station", "transit"),
            ("beginner-friendly guidance with no prerequisite knowledge", "skill_level"),
            ("family-friendly seating suitable for young children", "family_friendly"),
            ("senior-citizen friendly seating with comfortable back support", "ergonomic"),
            ("quiet, distraction-free environment for deep reading", "acoustic")
        ]
        desc, tag = rng.choice(constraints)

        intros = [
            f"We are planning an outing for {group} in {area}",
            f"Looking for weekend plans around {area} for a group of {group}",
            f"Can you suggest an afternoon plan in {area} for {group} people",
            f"Need weekend recommendations in {area} for {group} guests",
            f"Organizing a small gathering for {group} around {area}",
            f"Trying to arrange a get-together for {group} in {area}"
        ]

        budget_parts = [
            f"with a budget cap of ₹{budget} total",
            f"where our total spend is ₹{budget}",
            f"under ₹{budget} total",
            f"within a total budget of ₹{budget}",
            f"keeping total costs under ₹{budget}"
        ]

        constraint_phrases = [
            f"Mandatory requirement: {desc}.",
            f"Crucial condition: {desc} is essential.",
            f"We strictly require {desc}.",
            f"The venue must provide {desc}.",
            f"A non-negotiable for us is {desc}."
        ]

        u_text = f"{rng.choice(intros)} {rng.choice(budget_parts)}. {rng.choice(constraint_phrases)}"

        a_text = (
            f"Noted: {group} {'person' if group == 1 else 'people'} in {area} with a ₹{budget} total budget and a strict requirement for {desc}. "
            f"I'll factor in verified {tag} filters before searching our partner database. Would you prefer a daytime morning session or an afternoon slot?"
        )
        sys_prompt = "You are the FlowersForYou planner. Respect specific user constraints. Ask essential clarifying questions before tool search."
        return {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": u_text},
                {"role": "assistant", "content": a_text}
            ]
        }

    # -------------------------------------------------------------
    # 6. Multi-Turn Conversations & Corrections
    # -------------------------------------------------------------
    def generate_multi_turn_corrections(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        area1, area2 = rng.sample(CITIES_AREAS[city], 2)
        act1, act2 = rng.sample(ALL_ACTIVITIES, 2)
        b1 = rng.choice([400, 500, 600, 700, 800, 1000])
        b2 = rng.choice([800, 1000, 1200, 1500, 2000])
        timing1 = rng.choice(["Saturday morning", "Saturday afternoon", "Friday evening"])
        timing2 = rng.choice(["Sunday afternoon", "Sunday morning", "next weekend"])
        group = rng.choice([3, 4, 5, 6])
        flavor = rng.choice(["activity_and_area", "budget_and_timing", "group_expansion", "constraint_addition", "vibe_switch"])

        sys_prompt = "You are the FlowersForYou assistant. Gracefully handle user corrections and updated preferences without arguing or ignoring changes."

        if flavor == "activity_and_area":
            return {
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"I was thinking about doing {act1['name']} around {area1} with a budget of ₹{b1}."},
                    {"role": "assistant", "content": f"{act1['with_art'].capitalize()} in {area1} sounds wonderful! Are you planning for Saturday or Sunday, and will you be going solo or with friends?"},
                    {"role": "user", "content": f"Actually, change of plans—my friend wants {act2['with_art']} instead, and our budget is now ₹{b2}. Can we switch to {area2}?"},
                    {"role": "assistant", "content": f"Updated smoothly! I've switched your activity to {act2['with_art']}, adjusted the neighborhood to {area2}, and increased your budget to ₹{b2}. What time window works best for you both?"}
                ]
            }
        elif flavor == "budget_and_timing":
            return {
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Can we plan {act1['with_art']} in {area1} for {timing1}? Budget is around ₹{b2}."},
                    {"role": "assistant", "content": f"I can help with that! {act1['with_art'].capitalize()} around {area1} for {timing1} fits nicely within ₹{b2}. Should I look for morning or afternoon slots?"},
                    {"role": "user", "content": f"Hold on, my budget is actually tighter—keep it under ₹{b1}, and let's move it to {timing2} instead."},
                    {"role": "assistant", "content": f"Got it! Adjusted your budget cap down to ₹{b1} and moved your preferred time to {timing2}. Would you like me to look into options for {timing2} now?"}
                ]
            }
        elif flavor == "group_expansion":
            return {
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"I'm looking for a solo spot for {act1['with_art']} in {area1} under ₹{b1}."},
                    {"role": "assistant", "content": f"A solo session for {act1['name']} in {area1} is a great way to unwind. Are you looking for a morning or evening session?"},
                    {"role": "user", "content": f"Actually, {group} of my colleagues want to tag along now! We need a space for {group + 1} people in {area1}."},
                    {"role": "assistant", "content": f"No problem at all! I've updated the plan from a solo outing to a group of {group + 1} in {area1}. Would you like me to find venues that can accommodate group bookings?"}
                ]
            }
        elif flavor == "constraint_addition":
            c_text = rng.choice(["wheelchair step-free access", "strictly vegan culinary options", "pet-friendly seating for a dog", "a quiet low-noise environment"])
            return {
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Let's look into {act1['with_art']} around {area1} for this weekend."},
                    {"role": "assistant", "content": f"{act1['with_art'].capitalize()} in {area1} sounds lovely! What approximate budget are you aiming for?"},
                    {"role": "user", "content": f"Budget is around ₹{b1}. Also, I forgot to mention: one person in our group requires {c_text}. Please make sure the venue accommodates that."},
                    {"role": "assistant", "content": f"Important detail noted! I've added a strict filter for {c_text} in {area1} with a ₹{b1} budget. Are you looking at Saturday or Sunday?"}
                ]
            }
        else:
            act_outdoor = rng.choice(ACTIVITIES["active_outdoor"])
            act_relax = rng.choice(ACTIVITIES["relaxed_wellness"])
            return {
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"I was considering {act_outdoor['with_art']} in {area1}."},
                    {"role": "assistant", "content": f"{act_outdoor['with_art'].capitalize()} in {area1} sounds energizing! Do you have a specific time in mind?"},
                    {"role": "user", "content": f"To be honest, work was exhausting this week. Can we switch to something quiet and relaxing like {act_relax['with_art']} instead?"},
                    {"role": "assistant", "content": f"Completely understand—taking it easy is so important. Switched our focus from high energy to {act_relax['with_art']} in {area1}. Would you prefer Saturday afternoon or Sunday morning?"}
                ]
            }

    # -------------------------------------------------------------
    # 7. Undecided Users & Skip Onboarding (Fixes Context Amnesia)
    # -------------------------------------------------------------
    def generate_undecided_and_skip(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        is_skip = rng.random() < 0.55
        sys_prompt = "You are the FlowersForYou onboarding assistant. Respect the user's choice to skip onboarding."

        if is_skip:
            has_area = rng.random() < 0.70
            if has_area:
                area = rng.choice(CITIES_AREAS[city])
                budget = rng.choice([400, 500, 600, 750, 900, 1000, 1200, 1500])
                timing = rng.choice(["Saturday morning", "Saturday afternoon", "Sunday morning", "Sunday afternoon", "this weekend", "Friday evening"])

                hooks = [
                    "Skip the intro questionnaire please,", "I'd rather skip the profile questions,",
                    "No surveys for me today,", "Let's bypass onboarding,",
                    "Can we skip the setup questions?", "I don't feel like doing a survey right now,",
                    "Rather skip the profile quiz for now,", "Bypass the intro questions please,",
                    "Please skip the onboarding flow,", "Skip the questions for now—",
                    "Let's skip the introductory questionnaire,", "Bypassing the questionnaire,",
                    "Not interested in a questionnaire today,", "Can we bypass the setup steps?",
                    "Skip the onboarding steps,", "Prefer to bypass the setup quiz,",
                    "No questionnaire needed right now,", "Skip the survey please,"
                ]

                motives = [
                    f"let's just plan an outing around {area}.",
                    f"can we browse activities in {area} right away?",
                    f"let's jump straight to finding things to do in {area}.",
                    f"let's look into plans near {area} immediately.",
                    f"I want to see what's happening around {area} this weekend.",
                    f"help me pick something to do in {area} directly.",
                    f"let's dive directly into activities in {area}.",
                    f"let's find something fun to do in {area} right now.",
                    f"I just want recommendations around {area} directly.",
                    f"show me verified activities in {area} straight away.",
                    f"let's get straight to planning in {area}.",
                    f"help me discover weekend options in {area} right now.",
                    f"let's look up open sessions in {area} without the quiz.",
                    f"I want to explore what's available around {area} today."
                ]

                extras = [
                    f"Budget cap is roughly ₹{budget}.",
                    f"Looking for {timing}.",
                    f"Hoping to keep it under ₹{budget} for {timing}.",
                    f"Planning for {timing}.",
                    f"Upper limit is around ₹{budget}.",
                    "Attending solo.",
                    "Going out with a friend.",
                    "Planning for a small group.",
                    ""
                ]

                parts = [rng.choice(hooks), rng.choice(motives)]
                extra = rng.choice(extras)
                if extra:
                    parts.append(extra)
                u_text = " ".join(parts)

                # CRITICAL FIX: Assistant acknowledges stated area and DOES NOT ask for it again!
                a_text = rng.choice([
                    f"Understood—skipping the questions entirely! Let's get straight to planning for {area}. What is your approximate budget, and are you planning solo or with friends?",
                    f"Skipping the questions! Let's dive right into options for {area}. What spending limit would you like to keep in mind, and what timing works best for you?",
                    f"Got it, no surveys! Let's find great options around {area} immediately. Roughly what budget per person should I target?",
                    f"No problem, bypassing setup questions. Focusing straight on {area}—what kind of budget range should we look at?",
                    f"All set, skipping onboarding. Let's look into {area} right now! What budget and timing do you have in mind?",
                    f"Skipping onboarding! We'll head straight into {area} activities. Are you looking for morning or afternoon slots, and what budget should we keep?"
                ])
            else:
                hooks = [
                    "Skip the intro questions please,", "I'd rather skip the profile questionnaire,",
                    "Bypass the onboarding questions,", "No surveys for me today,",
                    "Can we bypass the onboarding quiz,", "Skip the setup questionnaire,",
                    "Prefer not to do a survey right now,"
                ]
                motives = [
                    f"let's find an activity in {city} right away.",
                    f"and jump straight into planning in {city}.",
                    f"let's browse plans around {city} immediately.",
                    f"let's dive straight into activities in {city}.",
                    f"and look at open options in {city} directly."
                ]
                u_text = f"{rng.choice(hooks)} {rng.choice(motives)}"
                a_text = rng.choice([
                    f"Understood—skipping the questions entirely! Let's get straight to planning. Which area around {city} do you want to explore, and what is your approximate budget?",
                    f"Skipping the questions! Which neighborhood in {city} are you thinking of, and what spending limit should we target?",
                    f"Got it, jumping straight to planning. Where in {city} should we look, and what is your approximate budget?"
                ])
        else:
            area = rng.choice(CITIES_AREAS[city])
            openers = [
                f"I'm looking for weekend plans around {area}, but",
                f"Stuck in my apartment in {area} because",
                f"Trying to plan something for Saturday in {area}, but",
                f"Looking ahead to Sunday near {area}, however",
                f"New to weekend planning around {area} and",
                f"Having a hard time choosing plans in {area} as"
            ]
            indecision_phrases = [
                "I really don't know what I enjoy doing on weekends.",
                "I'm completely undecided and have trouble figuring out what activity to pick.",
                "I have no clue what I'm in the mood for. Everything sounds either too tiring or boring.",
                "I feel torn between doing something creative or just relaxing.",
                "I'm completely undecided on whether I want an active day or a quiet one.",
                "I haven't developed any consistent hobbies yet.",
                "I have trouble making up my mind when there are too many options.",
                "I'm not sure what my tastes are when it comes to local outings."
            ]
            asks = [
                "What contrasting options would you suggest?",
                "Could you give me a few completely different options to pick from?",
                "What are two or three contrasting starting points?",
                "Could you propose a couple of different vibes to choose between?",
                "How would you suggest I break out of this indecision?"
            ]
            u_text = f"{rng.choice(openers)} {rng.choice(indecision_phrases)} {rng.choice(asks)}"
            act1 = rng.choice(ACTIVITIES["creative"])["with_art"]
            act2 = rng.choice(ACTIVITIES["social_games"])["with_art"]
            act3 = rng.choice(ACTIVITIES["relaxed_wellness"])["with_art"]
            a_text = (
                f"That's completely fine—we can discover what you enjoy together! Here are three contrasting options in {area}: "
                f"would you prefer {act1}, {act2}, or {act3}?"
            )
            sys_prompt = "You are the FlowersForYou onboarding assistant. When users are undecided or unsure of their tastes, offer two or three concrete contrasting activity choices rather than open-ended meta-questions."

        return {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": u_text},
                {"role": "assistant", "content": a_text}
            ]
        }

    # -------------------------------------------------------------
    # 8. Tool-Grounded Search & Verified Facts
    # -------------------------------------------------------------
    def generate_tool_grounded_search(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        area = rng.choice(CITIES_AREAS[city])
        cat = rng.choice(["creative", "culinary", "social_games", "active_outdoor"])
        act = rng.choice(ACTIVITIES[cat])
        venue = rng.choice(VENUES_CATALOG)
        price = rng.choice([350, 450, 550, 650, 750, 850, 950, 1200, 1400])
        slot1 = rng.choice(["Saturday 11:00 AM", "Saturday 3:00 PM", "Sunday 11:30 AM", "Sunday 4:00 PM"])
        slot2 = rng.choice(["Saturday 5:30 PM", "Sunday 2:00 PM", "Sunday 6:00 PM"])

        tool_payload = {
            "search_results": [
                {
                    "venue_name": venue,
                    "area": area,
                    "city": city,
                    "verified_price_inr": price,
                    "available_slots": [slot1, slot2],
                    "partner_status": "verified_partner",
                    "amenities": ["materials_provided", "guided_instruction"]
                }
            ]
        }
        query_intros = [
            "Are there any verified", "Please check verified partner listings for",
            "Looking for open slots for", "Can you query backend records for verified",
            "Show me verified availability for", "Do you have verified partners offering",
            "Can you search verified listings for", "Could you check live partner availability for",
            "Looking for verified sessions of", "Please run a partner search for"
        ]
        query_tails = [
            f"{act['name']} sessions available in {area} for this weekend under ₹{price + 200}?",
            f"{act['name']} in {area} with a budget cap of ₹{price + 200}.",
            f"{act['name']} around {area}. What verified options exist under ₹{price + 200}?",
            f"{act['name']} workshops in {area} around ₹{price + 200}?",
            f"{act['name']} in {area} below ₹{price + 200}.",
            f"{act['name']} near {area} under ₹{price + 200}?",
            f"{act['name']} around {area} with slots this Saturday or Sunday?",
            f"{act['name']} in {area} for two people under ₹{price + 200}?"
        ]
        u_text = f"{rng.choice(query_intros)} {rng.choice(query_tails)}"
        tool_text = json.dumps(tool_payload)
        a_text = (
            f"Yes! Based on verified search results, {venue} in {area} offers {act['with_art']} at ₹{price} per person with materials provided. "
            f"Verified open slots are currently {slot1} and {slot2}. Would you like me to share details on how to reserve one of these slots?"
        )
        sys_prompt = "You are the FlowersForYou planner. Ground recommendations in verified tool search results. Do not invent unverified perks, transit routes, or availability."
        return {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": u_text},
                {"role": "tool", "content": tool_text},
                {"role": "assistant", "content": a_text}
            ]
        }

    # -------------------------------------------------------------
    # 9. Booking Safety & Verification (Negative Constraint Training)
    # -------------------------------------------------------------
    def generate_booking_safety_and_verification(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        area = rng.choice(CITIES_AREAS[city])
        venue = rng.choice(VENUES_CATALOG)
        price = rng.choice([250, 350, 450, 500, 600, 750, 850, 1000, 1200, 1500])
        group = rng.choice([2, 3, 4, 5, 6, 8])
        time_slot = rng.choice(["10:30 PM", "11:00 PM", "11:30 PM", "midnight", "1:00 AM", "2:00 AM"])
        discount_pct = rng.choice([10, 15, 20, 25, 30, 40])
        slot_time = rng.choice(["7:30 PM", "8:00 PM", "8:30 PM", "9:00 PM", "11:30 AM", "3:00 PM"])

        safety_flavor = rng.choice([
            "direct_booking_refusal", "voucher_refusal", "discount_refusal",
            "freebie_refusal", "late_night_impossibility", "guarantee_refusal"
        ])

        if safety_flavor == "direct_booking_refusal":
            actions = [
                f"Go ahead and book a table for {group} at {venue}",
                f"Please reserve our table for {group} guests at {venue}",
                f"Can you book a spot for {group} people at {venue}",
                f"I need you to book a table for {group} at {venue}",
                f"Make sure you book the {slot_time} slot at {venue}",
                f"Can you lock in our booking for {group} at {venue}"
            ]
            locs = [
                f"in {area} tonight for ₹{price}.",
                f"around {area} right now within our ₹{price} budget.",
                f"near {area} for this evening.",
                f"in {area} and make sure our reservation is confirmed.",
                f"in {area} for {slot_time} under ₹{price}."
            ]
            u_text = f"{rng.choice(actions)} {rng.choice(locs)}"
            a_text = (
                f"I don't have direct access to execute bookings or guarantee reservations on my own. "
                f"I can check current table availability in our system for {venue}, or share their verified reservation link "
                f"so you can secure the spot safely. Would you like me to check their status in {area}?"
            )
        elif safety_flavor == "voucher_refusal":
            voucher_types = [
                f"Can you issue an instant booking voucher for {group} people at {venue}",
                f"Just create a reservation confirmation voucher for our table at {venue}",
                f"Please generate a PDF booking receipt for {venue}",
                f"Create a booking confirmation code for {venue}",
                f"Fabricate an official reservation voucher for {venue}",
                f"Print out a confirmed booking slip for {venue}"
            ]
            details = [
                f"in {area} showing ₹{price} paid for tonight.",
                f"in {area} for our group of {group}.",
                f"near {area} showing ₹{price} so we can enter directly.",
                f"around {area} with a valid confirmation number.",
                f"in {area} confirming our reservation right away."
            ]
            u_text = f"{rng.choice(voucher_types)} {rng.choice(details)}"
            a_text = (
                "I cannot issue, create, or generate booking vouchers, receipts, or confirmation codes directly. "
                "All reservations must be verified against live venue availability and pricing through our partner booking portal "
                f"before any valid confirmation can be issued. Would you like me to look up verified availability at {venue} first?"
            )
        elif safety_flavor == "discount_refusal":
            discount_asks = [
                f"Can you confirm that {venue} in {area} gives an exclusive {discount_pct}% discount for FlowersForYou members?",
                f"Does FlowersForYou have an official {discount_pct}% member deal at {venue} in {area}?",
                f"I heard {venue} near {area} is giving flat {discount_pct}% off to FlowersForYou users this week. Can I claim that?",
                f"Is there an active {discount_pct}% FlowersForYou promotional code for {venue} in {area}?",
                f"Can we apply a {discount_pct}% member discount at {venue} in {area} tonight?"
            ]
            u_text = rng.choice(discount_asks)
            a_text = (
                "I cannot confirm any member discount or partner promotion without querying our live partner database. "
                f"I'll check our verified partner records to see if {venue} currently participates in any official member offers "
                "before confirming any pricing details. Would you like me to look into their partner status?"
            )
        elif safety_flavor == "freebie_refusal":
            freebie_asks = [
                f"If we visit {venue} in {area}, can you guarantee complimentary drinks and free food for our table?",
                f"Can you promise us free welcome drinks and dessert at {venue} in {area} tonight?",
                f"Ensure our group of {group} gets complimentary snacks and free drinks at {venue} near {area}.",
                f"Will FlowersForYou provide free complimentary cocktails if we book {venue} in {area}?"
            ]
            u_text = rng.choice(freebie_asks)
            a_text = (
                "I cannot promise complimentary drinks, free desserts, or special perks unless they are verified in the venue's "
                f"active partner promotion records. I can check whether {venue} currently offers any verified partner promotions in {area}. "
                "Would you like me to check that?"
            )
        elif safety_flavor == "late_night_impossibility":
            act = rng.choice(ACTIVITIES["creative"])
            night_asks = [
                f"Find me an open studio for {act['name']} in {area} at {time_slot} tonight for ₹{price}.",
                f"Can we book {act['with_art']} in {area} at {time_slot} for ₹{price}?",
                f"Looking for an open {act['name']} workshop in {area} around {time_slot} tonight.",
                f"Can you find a workshop offering {act['name']} operating at {time_slot} in {area}?"
            ]
            u_text = rng.choice(night_asks)
            a_text = (
                f"Creative craft studios in {area} do not operate at {time_slot}, as workshops run during daytime and early evening hours. "
                f"I can search for daytime weekend slots within your ₹{price} budget, or recommend late-night cafes if you're looking to meet tonight. "
                "Which would you prefer?"
            )
        else:
            guarantee_asks = [
                f"Guarantee me a table at {venue} in {area} tonight without checking availability.",
                f"Lock in our reservation for {group} guests at {venue} in {area} right now 100% guaranteed.",
                f"Promise me that our table at {venue} in {area} will definitely be available without verification.",
                f"Secure a guaranteed spot for {group} at {venue} in {area} immediately."
            ]
            u_text = rng.choice(guarantee_asks)
            a_text = (
                "I cannot guarantee tables or reservations without checking real-time availability in our system. "
                f"All bookings must go through the venue's live booking channel. Would you like me to check verified availability at {venue} for your group?"
            )

        sys_prompt = "You are the FlowersForYou planner. Do not invent availability or prices. Never promise or confirm reservations, and never invent discounts."
        return {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": u_text},
                {"role": "assistant", "content": a_text}
            ]
        }

    # -------------------------------------------------------------
    # 10. Ambiguous Requests & Privacy Protection
    # -------------------------------------------------------------
    def generate_ambiguous_and_privacy(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        area = rng.choice(CITIES_AREAS[city])
        flavor = rng.choice(["privacy_card", "privacy_address", "privacy_password", "privacy_gov_id", "privacy_api_key", "vague_plan", "impossible_activity"])

        if flavor == "privacy_card":
            card_types = ["Visa", "Mastercard", "RuPay", "Amex"]
            card_type = rng.choice(card_types)
            c1, c2, c3, c4 = rng.randint(4000, 4999), rng.randint(1000, 9999), rng.randint(1000, 9999), rng.randint(1000, 9999)
            cvv = rng.randint(100, 999)
            card_intros = [
                f"Here is my {card_type} card {c1} {c2} {c3} {c4} with CVV {cvv} and expiry 12/28.",
                f"Please charge my {card_type} card {c1}-{c2}-{c3}-{c4}, CVV {cvv}, expiry 09/27.",
                f"Save my {card_type} card details: number {c1} {c2} {c3} {c4}, security code {cvv}, exp 05/29.",
                f"My card info is {c1} {c2} {c3} {c4} ({card_type}) with CVV {cvv}.",
                f"Keep my payment card {c1} {c2} {c3} {c4} on file with CVV {cvv} and expiry 11/28."
            ]
            card_tails = [
                f"Save this so you can auto-book any weekend workshops in {area}.",
                f"Use this to pay for all reservations in {area} automatically.",
                f"Store this in my profile for fast checkout around {area}.",
                f"Keep it on file so I don't have to enter it for plans in {area}."
            ]
            u_text = f"{rng.choice(card_intros)} {rng.choice(card_tails)}"
            a_text = (
                "For your privacy and security, I cannot accept or store sensitive financial details like credit card numbers or CVVs. "
                "FlowersForYou only records activity preferences, general neighborhoods, and budget ranges. When you choose an activity, "
                "you can complete your reservation directly on the venue's secure checkout page."
            )
        elif flavor == "privacy_gov_id":
            pan = f"{''.join(rng.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))}{rng.randint(1000, 9999)}{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"
            gov_intros = [
                f"My government tax PAN ID is {pan}.",
                f"Please link my tax identity document {pan} to my account.",
                f"Here is my national ID number: {pan}.",
                f"Save my PAN verification code {pan} in my profile."
            ]
            gov_tails = [
                f"Store this in my profile so local venues in {area} can verify my identity.",
                f"Keep this handy for future bookings around {area}.",
                f"Make sure this is recorded for reservations in {area}."
            ]
            u_text = f"{rng.choice(gov_intros)} {rng.choice(gov_tails)}"
            a_text = (
                "For your privacy and data security, I do not collect or store government identity documents like PAN cards or tax numbers. "
                f"FlowersForYou only maintains activity tastes and general location preferences around {area}. What kind of outing would you like to explore?"
            )
        elif flavor == "privacy_api_key":
            key_suffix = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=16))
            api_intros = [
                f"Here is my personal API authorization token 'ffy_sec_{key_suffix}'",
                f"My developer access token is 'sk_live_{key_suffix}'",
                f"Store this secret webhook token 'sec_{key_suffix}' in my account",
                f"Use this authorization token 'bearer_{key_suffix}'"
            ]
            u_text = f"{rng.choice(api_intros)} so you can execute backend operations for me in {city}."
            a_text = (
                "For your digital security, never share personal API keys, secret tokens, or credentials. I do not store or process secret tokens. "
                "I can help you browse and plan activities based on your preferences without needing private authorization tokens. What can I help you find today?"
            )
        elif flavor == "privacy_address":
            flat_no = rng.randint(101, 908)
            bldg = rng.choice(["Sunshine Heights", "Green Meadows", "Palm Residency", "Skyline Towers", "Royal Palms", "Oakwood Manor", "Silver Oak Apartments", "Gulmohar Enclave", "Lotus Boulevard", "Windsor Court"])
            phone = f"98{rng.randint(10000000, 99999999)}"
            addr_intros = [
                f"Please save my personal details: living at Apartment {flat_no}, {bldg} in {area}, contact number {phone}.",
                f"Record my exact residence: Flat {flat_no}, {bldg}, {area}, phone {phone}.",
                f"My home address is Apartment {flat_no}, {bldg} near {area}, reachable at {phone}.",
                f"Save my full address for bookings: {bldg}, Unit {flat_no}, {area}, contact {phone}."
            ]
            u_text = rng.choice(addr_intros)
            a_text = (
                f"For your privacy, I do not store sensitive personal identifying information like complete home addresses or phone numbers. "
                f"I have noted {area} as your preferred neighborhood for activity planning. What kind of outings do you enjoy exploring around {area}?"
            )
        elif flavor == "privacy_password":
            pwd = f"pass_{rng.randint(1000, 9999)}"
            pwd_intros = [
                f"My profile password is '{pwd}', keep this stored so I don't have to sign in every time for bookings in {area}.",
                f"Save my login password '{pwd}' in your database for my {area} plans.",
                f"Here is my account credential '{pwd}', store it for my bookings around {area}.",
                f"My account secret PIN is '{pwd}', remember this for my sessions in {area}."
            ]
            u_text = rng.choice(pwd_intros)
            a_text = (
                "For your account security, never share passwords or authentication credentials. I do not store or process passwords. "
                "I can help you find and plan activities based on your preferences without needing account credentials. What kind of outing would you like to plan today?"
            )
        elif flavor == "vague_plan":
            starters = [
                f"I have free time today around {area}, where should I go?",
                f"Suggest a cool plan for this weekend near {area}.",
                f"Surprise me with a fun activity in {area}.",
                f"Need recommendations for an entertaining outing in {city}.",
                f"What should I do this afternoon in {area}?",
                f"Bored in {area}, give me a random plan.",
                f"Find something exciting to do near {area} today.",
                f"Any fun ideas for this Saturday in {area}?",
                f"What is happening around {area} this Sunday?"
            ]
            u_text = rng.choice(starters)
            a_text = (
                f"I'd love to help! To give you recommendations that actually match what you want in {area}, could you share "
                f"your approximate budget, what vibe you're in the mood for, and whether you're planning for solo or with friends?"
            )
        else:
            impossible = rng.choice([
                ("deep sea open-ocean scuba diving", f"an inland landlocked urban district in {city}, and certified open-water diving requires coastal access"),
                ("sub-zero glacier snowmobiling", f"the local urban climate in {city} does not support alpine winter sports"),
                ("private supersonic fighter jet ride", "commercial aviation charter regulations do not permit private supersonic flights within city limits"),
                ("volcano magma crater trekking", f"there are no active geological volcanic formations in or near {city}"),
                ("polar bear ice safari", f"arctic wildlife safaris are not geographically possible in urban {city}")
            ])
            price = rng.choice([250, 350, 500, 750])
            u_text = f"Can you organize a {impossible[0]} session in {area} tomorrow morning for ₹{price}?"
            a_text = (
                f"{impossible[0].title()} is not available in {area} as it is {impossible[1]}. "
                f"If you're looking for an active morning in {area}, we could look into indoor climbing or sporting sessions within your ₹{price} budget. Would that interest you?"
            )

        sys_prompt = "You are the FlowersForYou planner. Protect user privacy and clarify ambiguous requests politely."
        return {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": u_text},
                {"role": "assistant", "content": a_text}
            ]
        }

    # -------------------------------------------------------------
    # 11. User Feedback & Recovery
    # -------------------------------------------------------------
    def generate_feedback_and_recovery(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        area = rng.choice(CITIES_AREAS[city])
        act = rng.choice(ALL_ACTIVITIES)

        issues = [
            ("way too loud and packed so we couldn't talk comfortably",
             "I'm sorry to hear that the acoustic environment was overwhelming! I've noted a strong preference for quiet, relaxed venues with good table spacing. Next time, would you prefer a peaceful tea lounge with private nooks, or a smaller boutique parlor that limits group bookings?"),
            ("lacking in guidance as the instructor didn't help beginners",
             "Thank you for sharing that feedback—I understand how frustrating it is when a beginner class lacks proper instruction. I'll make sure future recommendations prioritize small-cohort workshops with dedicated 1-on-1 mentorship. Would you like to try another craft style like candle making or linocut printing instead?"),
            ("disrupted by rain because there was no covered shelter nearby",
             "That's such a disappointment when weather cuts an outing short! I've added a note to prioritize venues with rain-protected pavilions or hybrid indoor-outdoor seating for future outings. Shall we look for a cozy covered venue for your next get-together?"),
            ("severely lacking in vegan options on their menu",
             "I apologize for that oversight! Having reliable dietary choices is essential. I've updated your profile with a strict filter for verified vegan-certified partner cafes in this area. Would you like to check out a dedicated plant-based bakery for your next visit?"),
            ("impossible to park near and we circled for almost an hour",
             "Finding parking should not add stress to an outing! I've recorded a priority preference for venues with dedicated parking or direct valet services, or spots located within a 5-minute walk of a metro station. Would you prefer metro-accessible venues for future plans?"),
            ("overcrowded with over 25 people crammed into one studio",
             "That completely defeats the purpose of an intimate hands-on workshop! I have noted a maximum group cap preference of 8–10 participants for creative classes. Would you be interested in exploring private studio sessions where attendance is strictly capped?"),
            ("overpriced with surprise cover fees not shown on the listing",
             "Unexpected pricing discrepancies are very frustrating. We only endorse partners who honor verified all-inclusive rates without hidden covers. I can look into verified specialty venues with fixed transparent pricing in this area if you'd like to try again."),
            ("poorly ventilated and stuffy throughout the afternoon",
             "I'm very sorry the room comfort was poor. I've updated your venue filters to ensure good climate control and airy indoor spaces for future sessions. Would you like to explore an outdoor covered terrace next time?"),
            ("provided with dull or broken tools during the workshop",
             "That's definitely not the quality standard our partners should provide. I have flagged this partner for quality review. Can I help you find another studio with top-rated materials for your next session?"),
            ("feeling rushed by the facilitators before we could finish",
             "A relaxing creative workshop should never feel hurried. I've updated your preferences to favor self-paced studios with extended open-hours. Would you like to try a drop-in pottery or painting lounge for your next outing?")
        ]
        desc, recovery = rng.choice(issues)
        openers = [
            f"The {act['name']} we visited in {area} was {desc}.",
            f"I went for {act['with_art']} around {area} and it was {desc}.",
            f"Our group tried {act['with_art']} in {area} yesterday, but it was {desc}.",
            f"Honest feedback on our {act['name']} in {area}: it was {desc}.",
            f"Just wanted to report that our visit for {act['name']} in {area} was {desc}.",
            f"Quick review of our {act['name']} outing in {area}: unfortunately it was {desc}."
        ]
        u_text = rng.choice(openers)
        sys_prompt = "You are the FlowersForYou assistant. Acknowledge user dissatisfaction empathetically, update preferences, and offer constructive next steps."
        return {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": u_text},
                {"role": "assistant", "content": recovery}
            ]
        }


def main():
    parser = argparse.ArgumentParser(description="FlowersForYou Repaired Dataset Generator (v2)")
    parser.add_argument("--batches", type=int, default=20, help="Number of batches to generate (default: 20)")
    parser.add_argument("--batch-size", type=int, default=500, help="Target records per batch (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed (default: 42)")
    parser.add_argument("--threshold", type=float, default=0.80, help="Diversity rejection threshold (default: 0.80)")
    args = parser.parse_args()

    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load held-out evaluation datasets for zero-leakage checking
    held_out_texts = []
    if EVAL_FILE.exists():
        with open(EVAL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                u = " ".join(m.get("content", "") for m in row.get("messages", []) if m.get("role") == "user")
                if u:
                    held_out_texts.append(u)
    if EVAL_EXPANDED_FILE.exists():
        with open(EVAL_EXPANDED_FILE, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                u = " ".join(m.get("content", "") for m in row.get("messages", []) if m.get("role") == "user")
                if u:
                    held_out_texts.append(u)

    print(f"Loaded {len(held_out_texts)} held-out evaluation prompts for zero-leakage protection.")

    engine = RepairedScenarioEngine(held_out_texts)
    diversity_index = DiversityIndex(threshold=args.threshold)

    total_rejections = defaultdict(int)
    all_accepted_records = []
    all_manifest_records = []

    cat_generators = {
        "first_time_onboarding": engine.generate_first_time_onboarding,
        "activity_discovery": engine.generate_activity_discovery,
        "structured_preference_extraction": engine.generate_structured_preference_extraction,
        "returning_users": engine.generate_returning_users,
        "constraints_handling": engine.generate_constraints_handling,
        "multi_turn_corrections": engine.generate_multi_turn_corrections,
        "undecided_and_skip_onboarding": engine.generate_undecided_and_skip,
        "tool_grounded_search": engine.generate_tool_grounded_search,
        "booking_safety_and_verification": engine.generate_booking_safety_and_verification,
        "ambiguous_and_privacy": engine.generate_ambiguous_and_privacy,
        "feedback_and_recovery": engine.generate_feedback_and_recovery,
    }

    print("=" * 70)
    print("FlowersForYou Phase 6C Repaired Dataset Generator (v2)")
    print(f"Target: {args.batches} batches x {args.batch_size} = {args.batches * args.batch_size} records")
    print(f"Diversity Threshold: Jaccard < {args.threshold} strictly enforced globally")
    print("=" * 70)

    t0_all = time.time()

    for b_idx in range(1, args.batches + 1):
        b_id = f"batch_{b_idx:02d}"
        b_file = BATCHES_DIR / f"{b_id}.jsonl"
        m_file = BATCHES_DIR / f"manifest_{b_id}.jsonl"
        r_file = BATCH_REPORTS_DIR / f"{b_id}_report.md"

        batch_rng = random.Random(args.seed + b_idx * 1000)
        batch_records = []
        manifest_records = []
        batch_rejections = defaultdict(int)

        for cat, quota in CATEGORY_BATCH_QUOTAS.items():
            gen_fn = cat_generators[cat]
            accepted = 0
            attempts = 0
            max_attempts = quota * 200

            while accepted < quota and attempts < max_attempts:
                attempts += 1
                cand = gen_fn(batch_rng)
                u_text = get_conversation_user_text(cand.get("messages", []))
                has_tool = any(m.get("role") == "tool" for m in cand.get("messages", []))

                sys_p = next((m.get("content", "") for m in cand.get("messages", []) if m.get("role") == "system"), "")
                u_p = next((m.get("content", "") for m in cand.get("messages", []) if m.get("role") == "user"), "")
                last_assist = next((m.get("content", "") for m in reversed(cand.get("messages", [])) if m.get("role") == "assistant"), "")
                eval_res = evaluate_response(
                    response=last_assist,
                    user_prompt=u_p,
                    system_prompt=sys_p,
                    has_tool_result=has_tool
                )
                if not eval_res.get("safe", True):
                    batch_rejections["evaluator_safety"] += 1
                    continue

                # 2. Strict Zero Leakage against held-out benchmarks
                if engine.is_eval_leak(u_text):
                    batch_rejections["eval_leakage"] += 1
                    continue

                # 3. Global Near-Duplicate Check (Jaccard < threshold across ALL accepted records)
                if diversity_index.is_near_duplicate(u_text):
                    batch_rejections["near_duplicate"] += 1
                    continue

                # Accept candidate
                diversity_index.add(u_text, len(all_accepted_records) + len(batch_records) + 1)
                accepted += 1
                ex_id = f"ffy_v2_{b_idx:02d}_{len(batch_records) + 1:03d}"
                batch_records.append(cand)
                manifest_records.append({
                    "id": ex_id,
                    "source_type": "synthetic",
                    "category": cat,
                    "batch_id": b_id,
                    "consent_status": "not_applicable_synthetic",
                    "dialogue_turns": len(cand.get("messages", [])),
                    "has_tool": has_tool,
                })

            if accepted < quota:
                raise RuntimeError(f"Category '{cat}' in {b_id} reached only {accepted}/{quota} after {attempts} attempts!")

        for k in total_rejections:
            total_rejections[k] += batch_rejections[k]

        # Write immutable batch
        with open(b_file, "w", encoding="utf-8") as f:
            for r in batch_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(m_file, "w", encoding="utf-8") as f:
            for m in manifest_records:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        # Write batch report
        report_text = f"""# FlowersForYou Batch Report (v2): {b_id.upper()}
- **Batch ID:** `{b_id}`
- **Accepted Records:** {len(batch_records)}
- **Validation Status:** PASS
- **Rejections:** {dict(batch_rejections)}
"""
        r_file.write_text(report_text, encoding="utf-8")
        print(f"[PASS] {b_id}: 500 records | Near-dups rejected: {batch_rejections['near_duplicate']} | Leakage rejected: {batch_rejections['eval_leakage']}")

        all_accepted_records.extend(batch_records)
        all_manifest_records.extend(manifest_records)

    # Compile final repaired dataset
    print("\n" + "=" * 70)
    print("Compiling Repaired v2 Dataset...")
    print("=" * 70)
    with open(EXPANDED_TRAIN_V2, "w", encoding="utf-8") as f:
        for r in all_accepted_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(MANIFEST_V2, "w", encoding="utf-8") as f:
        for m in all_manifest_records:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0_all
    print(f"Compiled: {EXPANDED_TRAIN_V2} ({len(all_accepted_records)} records)")
    print(f"Manifest: {MANIFEST_V2} ({len(all_manifest_records)} records)")
    print(f"Total Rejections during Generation: {dict(total_rejections)}")
    print(f"Completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
