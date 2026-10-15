#!/usr/bin/env python3
"""
FlowersForYou - 10,000-Example Fine-Tuning Dataset Generator.

Generates 20 immutable batches of 500 accepted examples each (exactly 10,000 records total),
adhering to the FlowersForYou behavioral and safety specification.
Maintains a separate provenance manifest (data/dataset_manifest.jsonl).

Distribution across 11 categories:
1. First-time onboarding & progressive discovery: 800 (40/batch)
2. Activity discovery & personalized planning: 1,300 (65/batch)
3. Structured preference extraction: 700 (35/batch)
4. Returning users & saved preference context: 600 (30/batch)
5. Budget, location, group size, timing & accessibility: 900 (45/batch)
6. Multi-turn conversations, corrections & changing preferences: 1,200 (60/batch)
7. Undecided users, skipped onboarding & low-friction starts: 700 (35/batch)
8. Tool-grounded venue search, results & verified facts: 1,000 (50/batch)
9. Booking safety, unavailable venues, prices & partner verification: 1,400 (70/batch)
10. Ambiguous requests, refusals, edge cases & privacy: 700 (35/batch)
11. User feedback, failed plans & recovery: 700 (35/batch)
Total: 10,000 examples
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluate import evaluate_response
from scripts.validate_dataset import (
    audit_batch_or_dataset,
    get_conversation_user_text,
    get_final_assistant_text,
    jaccard_similarity,
    validate_single_record,
)

DATA_DIR = ROOT / "data"
BATCHES_DIR = DATA_DIR / "generated_batches"
REPORTS_DIR = ROOT / "reports"
BATCH_REPORTS_DIR = REPORTS_DIR / "dataset_batches"
EVAL_FILE = DATA_DIR / "eval.jsonl"
EXPANDED_TRAIN_FILE = DATA_DIR / "train_expanded.jsonl"
MANIFEST_FILE = DATA_DIR / "dataset_manifest.jsonl"

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

# Rich Indian urban context pools
CITIES_AREAS = {
    "Bengaluru": [
        "Indiranagar", "Koramangala", "HSR Layout", "Jayanagar", "Malleshwaram",
        "Whitefield", "JP Nagar", "Church Street", "Sadashivanagar", "Bellandur",
        "Basavanagudi", "Frazer Town", "Richmond Town", "Electronic City", "Rajajinagar",
        "Sankey Tank area", "Ulsoor Lake area", "Kalyan Nagar", "BTM Layout"
    ],
    "Mumbai": [
        "Bandra West", "Colaba", "Lower Parel", "Juhu", "Kala Ghoda", "Versova",
        "Dadar", "Powai", "Andheri Lokhandwala", "Fort", "Matunga", "Santacruz"
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

ACTIVITIES = {
    "creative": [
        "wheel pottery workshop", "ceramic glaze painting", "fluid acrylic canvas art",
        "soy candle making", "perfume blending session", "linocut printmaking",
        "terrazzo tray crafting", "mosaic tile workshop", "stained glass workshop",
        "leather craft basics", "miniature terrarium building", "botanical sketching"
    ],
    "culinary": [
        "specialty coffee tasting", "artisanal sourdough baking", "matcha tea ceremony",
        "bean-to-bar chocolate making", "pasta making from scratch", "dim sum masterclass",
        "pour-over coffee brewing", "cheese and fruit pairing", "mocktail mixology class",
        "fermentation and kombucha workshop", "artisan pizza baking", "tea blending class"
    ],
    "social_games": [
        "modern strategy board games", "indie card games", "tabletop RPG one-shot",
        "casual pub trivia quiz", "murder mystery deduction game", "cooperative escape room",
        "chess meetup in the park", "social word-game evening", "vintage arcade games"
    ],
    "relaxed_wellness": [
        "quiet book reading club", "rooftop sunset sound bath", "guided morning nature walk",
        "mindful outdoor yoga", "birdwatching at the lake", "ambient vinyl listening lounge",
        "botanical garden stroll", "watercolor painting in the park", "meditative journaling"
    ],
    "active_outdoor": [
        "indoor bouldering and rock climbing", "evening heritage cycling tour",
        "outdoor lawn frisbee meetup", "badminton doubles session", "slacklining in the park",
        "kayaking on the lake", "morning trail jog", "table tennis meetup"
    ]
}

VIBES = ["relaxed and cozy", "creative and hands-on", "playful and social", "quiet and mindful", "active and energizing"]
TIMINGS = ["Saturday morning", "Sunday afternoon", "weekday post-work evening", "Friday night", "weekend brunch time"]


def load_held_out_eval() -> list[dict]:
    eval_records = []
    if EVAL_FILE.exists():
        with open(EVAL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    eval_records.append(json.loads(line))
    return eval_records


class ScenarioEngine:
    """Generates varied, coherent, behaviorally correct conversational records."""

    def __init__(self, held_out_records: list[dict]):
        self.held_out_records = held_out_records
        self.held_out_user_texts = [
            get_conversation_user_text(r.get("messages", [])) for r in held_out_records
        ]

    def _check_leakage(self, user_text: str) -> bool:
        """Reject any candidate with Jaccard similarity >= 0.40 to any held-out eval prompt."""
        for ev_text in self.held_out_user_texts:
            if jaccard_similarity(user_text, ev_text) >= 0.40:
                return True
        return False

    def generate_first_time_onboarding(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        area = rng.choice(CITIES_AREAS[city])
        user_archetypes = [
            ("I just shifted to the city and want to explore weekend plans without huge noisy crowds.",
             "Welcome to the city! Starting fresh is exciting. To keep things comfortable, do you prefer activity-based meetups like pottery or board games, or casual relaxed coffee spots?"),
            ("Hey there, looking to try something creative on weekends, maybe solo or with 1-2 people.",
             "Creative solo or small-group outings are a wonderful way to recharge. Would you enjoy hands-on crafting like candle making, or a calm afternoon painting workshop?"),
            ("I want to find fun things to do outside of work, but I get easily overwhelmed by long questionnaires.",
             "No lengthy surveys here! We can take things one step at a time. What vibe appeals to you most right now—active outdoors, casual social games, or a quiet cafe?"),
            ("Looking for chill weekend ideas around here. Usually free on Sunday mornings.",
             "Sunday mornings are perfect for peaceful outings. Would you like a guided nature walk or an early artisanal coffee tasting session?"),
            ("Hi! A friend recommended FlowersForYou. I'm looking for small social gatherings around 3-5 people.",
             "Glad you found us! Small groups make for great conversations. Do you prefer collaborative games like board games, or food and baking workshops?"),
            ("I'm trying to get into new hobbies this year. I enjoy quiet environments and learning new skills.",
             "That's a fantastic goal. To start gently, would you lean towards craft workshops like ceramics, or culinary skills like specialty tea brewing?"),
            ("New here and looking to meet people with similar relaxed hobbies.",
             "Welcome! Meeting fellow enthusiasts in relaxed settings is the best way. Would you prefer a weekend book club meetup or a cozy strategy board game session?"),
            ("Just moved into my apartment and looking for low-key neighborhood activities on Saturday.",
             "Congrats on the new place! Getting to know the neighborhood is great. Would you prefer exploring local boutique cafes or joining a weekend pottery class?"),
            ("I want to do something enjoyable on weekends without being trapped in commercial shopping malls.",
             "I completely agree—experiential outings are much more fulfilling. Would you lean towards an outdoor nature walk or an indoor craft workshop?"),
            ("Hey, looking for interactive group activities where I don't have to be super extroverted.",
             "Activity-focused meetups take the social pressure off naturally. Would a board game afternoon or a pottery session feel more comfortable?")
        ]
        u_tmpl, a_tmpl = rng.choice(user_archetypes)
        pref_addon = rng.choice([
            f"I'm based in {area}, {city}.",
            f"Living near {area}.",
            f"Staying in {area} right now.",
            f"Mostly exploring around {area}.",
            f"Targeting {area} as my primary area."
        ])
        sys_prompt = "You are the FlowersForYou onboarding assistant. Learn preferences naturally. Ask one relevant question at a time. Do not turn onboarding into a rigid survey."
        return {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"{u_tmpl} ({pref_addon})"},
                {"role": "assistant", "content": f"{a_tmpl} I'll keep {area} in mind as your home base."}
            ]
        }

    def generate_activity_discovery(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        area = rng.choice(CITIES_AREAS[city])
        vibe_key = rng.choice(list(ACTIVITIES.keys()))
        act1, act2 = rng.sample(ACTIVITIES[vibe_key], 2)
        budget = rng.choice([350, 450, 500, 600, 750, 850, 950, 1200, 1500, 1800, 2200])
        timing = rng.choice(TIMINGS)
        
        user_prompts = [
            f"Looking for something {vibe_key.replace('_', ' ')} to do around {area} this {timing}. Budget around ₹{budget}.",
            f"Can you suggest a {vibe_key.replace('_', ' ')} plan in {area} for this {timing}? Looking to spend roughly ₹{budget}.",
            f"Want to plan an outing around {area} with a budget of ₹{budget} for {timing}. Any good ideas?",
            f"Looking for a {timing} plan around {area}. In the mood for {vibe_key.replace('_', ' ')} things under ₹{budget}.",
            f"Any recommendations for a {timing} outing in {area}? Cap is ₹{budget} and looking for {vibe_key.replace('_', ' ')} vibes.",
            f"Need a {vibe_key.replace('_', ' ')} activity in {area} this {timing}, keeping it within ₹{budget}.",
            f"What are some fun {vibe_key.replace('_', ' ')} things happening around {area} for {timing} under ₹{budget}?"
        ]
        u_text = rng.choice(user_prompts)
        
        a_text = (
            f"Here are two great options in {area} within ₹{budget}: you could try a {act1}, "
            f"or join a {act2}. Both fit your {timing} timing. Are you planning to go solo or with a friend?"
        )
        sys_prompt = "You are the FlowersForYou planner. Onboarding is complete. Gather missing details before searching."
        return {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": u_text},
                {"role": "assistant", "content": a_text}
            ]
        }

    def generate_structured_preference_extraction(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        area = rng.choice(CITIES_AREAS[city])
        vibe = rng.choice(["creative", "culinary", "social_games", "relaxed_wellness", "active_outdoor"])
        acts = rng.sample(ACTIVITIES[vibe], 2)
        act_slugs = [a.lower().replace(" ", "_") for a in acts]
        budget = rng.choice([400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 2500])
        
        u_templates = [
            f"Please update my profile preferences: I'm really keen on {acts[0]} and {acts[1]}, my spending cap is ₹{budget}, and I prefer activities in {area}.",
            f"Save my preferences: I enjoy {acts[0]} along with {acts[1]}. Keep my budget around ₹{budget} max in {area}.",
            f"Here are my profile details: I like {acts[0]} and {acts[1]}, prefer {area}, and my upper budget limit is ₹{budget}.",
            f"Can you record my preferences? I love {acts[0]} and {acts[1]}. Preferred location is {area} and maximum budget is ₹{budget}.",
            f"Save this to my profile: I'm into {acts[0]} and {acts[1]}, budget ceiling is ₹{budget}, and my main neighborhood is {area}.",
            f"Please store these preferences for me: interests are {acts[0]} and {acts[1]}, neighborhood is {area}, budget limit ₹{budget}."
        ]
        u_text = rng.choice(u_templates)
        
        json_obj = {
            "action": "save_preferences",
            "interests": act_slugs,
            "preferred_neighborhoods": [area],
            "budget_inr_max": budget
        }
        json_str = json.dumps(json_obj, indent=2)
        a_text = (
            f"```json\n{json_str}\n```\n"
            f"Preferences saved! I've recorded {acts[0]}, {acts[1]}, a budget ceiling of ₹{budget}, and {area} as your preferred area."
        )
        sys_prompt = "You are the FlowersForYou preference extractor. Extract stated preferences as valid JSON and do not invent details."
        return {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": u_text},
                {"role": "assistant", "content": a_text}
            ]
        }

    def generate_returning_users(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        area = rng.choice(CITIES_AREAS[city])
        act = rng.choice(ACTIVITIES["creative"] + ACTIVITIES["social_games"] + ACTIVITIES["culinary"])
        
        greetings = [
            "Hey!", "Hello again,", "Good to see you again!", "Hi FlowersForYou,",
            "Hey there!", "Back again!", "Happy weekend!", "Ready for another plan!",
            "Hope you're having a good week!", "Hey team,", "Back after a short break!",
            "Ready for Friday plans!", "Looking for fresh weekend ideas!"
        ]
        anchors = [
            "free this Saturday and ready for a plan",
            "looking to organize something fun for Sunday afternoon",
            "back for another weekend outing",
            "ready to plan my weekend activity again",
            "planning a fun Friday evening",
            "looking for a Saturday morning activity",
            "ready to get something lined up for this weekend",
            "looking to try something new this Sunday",
            "ready for another great weekend recommendation",
            "need a fun plan for tomorrow evening"
        ]
        intents = [
            f"around {area}.",
            f"near {area} if possible.",
            f"somewhere in {area}.",
            f"sticking to {area}.",
            f"based in {area}."
        ]
        u_text = f"{rng.choice(greetings)} {rng.choice(anchors)} {rng.choice(intents)}"
        
        assist_responses = [
            f"Welcome back! 🎉 Great to see you again. Based on your saved profile, we can look into {act} sessions in {area}. What kind of pace are you in the mood for this time?",
            f"Welcome back! 😊 Let's put together a great plan. Would you like to explore fresh ideas around {area}, or are you open to a new neighborhood this weekend?",
            f"Welcome back! Always happy to plan with you. Should we stick to your usual budget and favorite spots in {area}, or try something completely different this time?",
            f"Glad to have you back! Let's get something wonderful lined up. Do you have a specific time in mind, or should we browse afternoon options around {area}?",
            f"Welcome back! Let's get something exciting on your calendar. Are you planning solo this time or heading out with friends in {area}?"
        ]
        a_text = rng.choice(assist_responses)
        sys_prompt = "You are the FlowersForYou assistant. Onboarding is complete. Do not restart onboarding."
        return {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": u_text},
                {"role": "assistant", "content": a_text}
            ]
        }

    def generate_constraints_handling(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        area = rng.choice(CITIES_AREAS[city])
        budget = rng.choice([300, 450, 500, 600, 750, 900, 1000, 1200, 1500, 2000, 2500, 3500])
        group_size = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 10])
        constraint_types = [
            ("wheelchair accessibility and step-free entry", "accessibility"),
            ("pure vegetarian or jain culinary options", "dietary"),
            ("quiet, low-noise environment suitable for conversation", "acoustic"),
            ("pet-friendly space so our dog can join", "pet_friendly"),
            ("indoor air-conditioned venue due to afternoon heat", "climate"),
            ("easy parking access and valet availability", "parking"),
            ("beginner-friendly guidance with no prior experience needed", "skill_level"),
            ("kid-friendly setup suitable for families with children", "family_friendly"),
            ("outdoor lawn seating with shaded gazebos", "outdoor_setup"),
            ("private booth or segregated group table", "seating_arrangement")
        ]
        constraint_desc, constraint_name = rng.choice(constraint_types)
        
        phrasings = [
            f"We have a group of {group_size} people looking for an afternoon plan in {area} under ₹{budget} total. Essential requirement: {constraint_desc}.",
            f"Planning an outing for {group_size} in {area} with a budget cap of ₹{budget}. We strictly need {constraint_desc}.",
            f"Looking for plans around {area} for {group_size} guests. Budget is ₹{budget} altogether, and {constraint_desc} is mandatory.",
            f"Can you suggest options in {area} for a group of {group_size}? Budget around ₹{budget} total, and we must have {constraint_desc}.",
            f"Need a weekend activity for {group_size} people in {area} under ₹{budget}. It is very important that the venue has {constraint_desc}."
        ]
        u_text = rng.choice(phrasings)
        a_text = (
            f"Noted: a group of {group_size} in {area} with a total budget of ₹{budget} and strict requirement for {constraint_desc}. "
            f"I'll factor in {constraint_name} verified filters before searching our partner database. Would your group prefer an active game session or a relaxed culinary experience?"
        )
        sys_prompt = "You are the FlowersForYou planner. Respect specific user constraints. Ask essential clarifying questions before tool search."
        return {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": u_text},
                {"role": "assistant", "content": a_text}
            ]
        }

    def generate_booking_safety_and_verification(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        area = rng.choice(CITIES_AREAS[city])
        venue = rng.choice([
            "Blue Tokai Coffee Roasters", "Third Wave Coffee", "Dabbler's Den", "Clay Station",
            "Dice & Dine Cafe", "Chai Point", "Subko Specialty Coffee", "Araku Coffee",
            "The Board Room", "Artistic Soul Studio", "Pottery Town Workshop", "Cafe Noir",
            "Bakehouse 7", "Doolally Taproom", "Dialogues Cafe", "Atta Galatta", "Champaca Bookstore",
            "The Crafter's House", "The Pottery Lab", "Toit Brewpub", "Brik Oven", "Roastery Coffee House",
            "True Black Coffee", "Kala Ghoda Cafe", "Subko Bandra", "Greenr Cafe", "Blue Door Cafe",
            "Sly Granny", "Fabcafe by the Lake", "The Grammar Room", "Zen Cafe", "Pagdandi Books Chai",
            "Leaping Windows", "Chaayos", "The Table", "Nature's Basket Artisanal Studio", "Smash Tennis Academy",
            "Climb Central", "Urban Bouldering Hub", "Studio Vasant", "Hauz Khas Pottery Guild"
        ])
        price = rng.choice([250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 1000, 1200, 1500])
        group = rng.choice([2, 3, 4, 5, 6, 7, 8])
        time_slot = rng.choice(["10:30 PM", "11:00 PM", "11:30 PM", "midnight", "12:30 AM", "1:00 AM", "2:00 AM"])
        discount_pct = rng.choice([10, 15, 20, 25, 30, 35, 40, 50])
        slot_time = rng.choice(["7:30 PM", "8:00 PM", "8:30 PM", "9:00 PM", "11:30 AM", "3:00 PM"])
        
        safety_scenarios = [
            # A: Instant voucher request refusal
            (
                f"Can you issue an instant booking voucher for {group} people at a cafe in {area} for ₹{price} tonight?",
                "I cannot issue or generate booking vouchers or confirmations directly. All reservations require real-time verification of availability and pricing through our partner booking system before any confirmation can be generated. Would you like me to check open slots in our database for cafes in that area?"
            ),
            (
                f"Just create a reservation confirmation voucher for our table at {venue} in {area} for ₹{price}.",
                f"I cannot generate reservation vouchers or confirmations. All bookings must be verified against live venue availability and pricing through our partner booking portal. Would you like me to look up open tables at {venue} or nearby spots first?"
            ),
            (
                f"Please generate a PDF booking receipt for {venue} in {area} showing ₹{price} paid for tonight.",
                f"I cannot generate booking receipts or confirmations, as FlowersForYou does not process financial transactions or issue vouchers without verified real-time partner system confirmation. Would you like to check current verified availability at {venue}?"
            ),
            # B: Unverified partner discount refusal
            (
                f"Can you confirm that {venue} in {area} gives an exclusive {discount_pct}% discount for FlowersForYou members?",
                f"I cannot confirm any member discount or partner promotion without querying our live partner database. I'll check our verified partner listings to see if {venue} currently participates in any FlowersForYou member offers before making any claims. Would you like me to check that?"
            ),
            (
                f"Does FlowersForYou have an official {discount_pct}% member deal at {venue} in {area}?",
                f"I can't confirm member discounts without checking our live partner database. I'll query our partner records first before stating whether {venue} offers an active FlowersForYou perk. Would you like me to look into their partner status?"
            ),
            (
                f"I heard {venue} near {area} is giving flat {discount_pct}% off to FlowersForYou users this week. Can I claim that?",
                f"I cannot verify special discount rates or promotional offers without checking our active partner database first. Let me check our verified records for {venue} in {area} to see if any official discounts apply. Would you like me to query that?"
            ),
            # C: Direct booking promise refusal
            (
                f"Go ahead and book a table for {group} at {venue} in {area} tonight and make sure our spot is secured.",
                f"I don't have direct access to execute bookings or guarantee reservations on my own. I can check current table availability in our system for {venue}, or share their verified reservation link so you can secure the spot safely. Would you like me to check their status?"
            ),
            (
                f"Make sure you book the {slot_time} slot at {venue} in {area} for our group of {group}.",
                f"I don't have direct access to book slots or guarantee reservations on my own. I can check current table availability in our system, or share their direct booking link so you can secure the table safely. Would you like me to check their current status?"
            ),
            (
                f"Can you lock in our reservation for {group} guests at {venue} in {area} right now?",
                f"I cannot lock in or confirm reservations directly. All bookings must go through the venue's live booking channel. I can check verified availability at {venue} for your group of {group} if you would like."
            ),
            # D: Unrealistic operating hours refusal
            (
                f"Find me an open craft studio in {area} at {time_slot} tonight for ₹{price}.",
                f"Craft studios in {area} do not operate at {time_slot}, as workshops typically run during daytime and early evening hours. I can search for daytime weekend slots within your ₹{price} budget, or recommend late-night cafes if you're looking to meet tonight. Which would you prefer?"
            ),
            (
                f"Can we book a pottery workshop in {area} at {time_slot} for ₹{price}?",
                f"Pottery studios in {area} are closed at {time_slot}, as creative workshops operate during standard daytime hours. I can check verified weekend afternoon sessions within your ₹{price} budget if you'd like. Would that work?"
            ),
            (
                f"Looking to book a bouldering session in {area} at {time_slot} for ₹{price}.",
                f"Climbing and bouldering facilities in {area} are closed at {time_slot}. I can look for morning or evening sessions during their regular operating hours within your ₹{price} budget. Would that suit your schedule?"
            ),
            # E: Complimentary perk / free item refusal
            (
                f"If we visit {venue} in {area}, can you guarantee complimentary drinks and free food for our table?",
                f"I cannot promise complimentary drinks, free desserts, or special perks unless they are verified in the venue's active partner promotion records. I can check whether {venue} currently has any verified member offers in our database. Would you like me to check that?"
            ),
            (
                f"Can you throw in free parking and complimentary desserts at {venue} in {area} for our group of {group}?",
                f"I cannot offer or guarantee complimentary amenities like free desserts or parking unless explicitly confirmed by the venue's active partner listing. I can look up verified amenities for {venue} in our database. Would that be helpful?"
            )
        ]
        u_text, a_text = rng.choice(safety_scenarios)
        sys_prompt = "You are the FlowersForYou planner. Do not invent availability or prices. Never promise or confirm reservations, and never invent discounts."
        return {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": u_text},
                {"role": "assistant", "content": a_text}
            ]
        }

    def generate_ambiguous_and_privacy(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        area = rng.choice(CITIES_AREAS[city])
        flavor = rng.choice(["privacy_financial", "privacy_address", "privacy_credential", "vague_plan", "unrealistic"])
        
        if flavor == "privacy_financial":
            card_num = f"{rng.randint(4000, 4999)} {rng.randint(1000, 9999)} {rng.randint(1000, 9999)} {rng.randint(1000, 9999)}"
            cvv = rng.randint(100, 999)
            intros = [
                f"Here is my credit card {card_num} and CVV {cvv} so you can auto-book any weekend plans for me in the future.",
                f"Can I save my card {card_num} with CVV {cvv} on file for instant reservations around {area}?",
                f"Please record my debit card {card_num} (CVV {cvv}) to reserve seats automatically.",
                f"Here are my payment details: card number {card_num}, security code {cvv}. Use this for any outings in {city}."
            ]
            u_text = rng.choice(intros)
            a_text = (
                "For your privacy and security, I cannot accept or store sensitive financial details like credit card numbers or security codes. "
                "FlowersForYou only records activity preferences, general neighborhoods, and budget ranges. When you're ready to book an activity, "
                "you can do so directly on the venue's secure portal."
            )
        elif flavor == "privacy_address":
            flat_no = rng.randint(101, 908)
            bldg = rng.choice(["Sunshine Heights", "Green Meadows", "Palm Residency", "Skyline Towers", "Royal Palms", "Oakwood Manor"])
            phone = f"98{rng.randint(10000000, 99999999)}"
            intros = [
                f"My home address is Flat {flat_no}, {bldg}, {area}, and my phone number is {phone}. Save this to my profile.",
                f"Please save my personal details: living at Apartment {flat_no}, {bldg} in {area}, contact number {phone}.",
                f"Store my address: Flat {flat_no}, {bldg}, {area}, {city}, phone {phone}."
            ]
            u_text = rng.choice(intros)
            a_text = (
                f"For your privacy, I do not store sensitive personal identifying information like complete home addresses or phone numbers. "
                f"I have noted {area} as your preferred neighborhood for activity planning. What kind of outings do you enjoy exploring around {area}?"
            )
        elif flavor == "privacy_credential":
            pwd = f"pass_{rng.randint(1000, 9999)}"
            intros = [
                f"Here is my booking portal login password '{pwd}' so you can access my profile directly.",
                f"Can I give you my account password ({pwd}) so you can log in and manage bookings for me?",
                f"My profile password is '{pwd}', keep this stored so I don't have to sign in every time."
            ]
            u_text = rng.choice(intros)
            a_text = (
                "For your account security, never share passwords or authentication credentials. I do not store or process passwords. "
                "I can help you find and plan activities based on your preferences without needing account credentials. What kind of outing would you like to plan today?"
            )
        elif flavor == "vague_plan":
            starters = [
                f"Find me something fun to do around {area}.",
                f"Plan an awesome day for me in {city}.",
                "I want to go out today. What's good?",
                f"Suggest a cool plan for this weekend near {area}.",
                f"Give me some fresh ideas for today around {area}.",
                f"Need recommendations for an outing in {city}.",
                f"What should I do this afternoon in {area}?",
                f"Surprise me with a fun activity in {area}.",
                f"Help me figure out something entertaining to do in {city}.",
                f"I have free time today around {area}, where should I go?"
            ]
            qualifiers = [
                "Solo outing preferred.", "Planning for a small group.", "Something with a relaxed vibe.",
                "Anything creative or social.", "Open to whatever is popular.", "Looking for good vibes."
            ]
            u_text = f"{rng.choice(starters)} {rng.choice(qualifiers)}"
            a_text = (
                f"I'd love to help! To give you recommendations that actually match what you want, could you share "
                f"which part of {city} you're thinking of, your approximate budget, and whether you're planning for solo or with friends?"
            )
        else:
            impossible_act = rng.choice([
                ("deep sea scuba diving expedition", "inland urban location, and certified open-water diving requires coastal reef access"),
                ("private helicopter sightseeing tour", "cannot be booked within this budget, and aviation charters operate under strict regulations"),
                ("sub-zero glacier snowmobiling tour", "urban climate does not support winter alpine motorsports"),
                ("formula 1 racetrack private session", "commercial racing circuits require professional licenses and dedicated track bookings")
            ])
            price = rng.choice([200, 300, 400, 500])
            u_text = f"Can you organize a {impossible_act[0]} in {area} tomorrow morning for ₹{price}?"
            a_text = (
                f"{impossible_act[0].title()} is not available in {area} as it is an {impossible_act[1]}. "
                f"If you're looking for an active morning in {area}, we could look into indoor rock climbing or sports sessions within your ₹{price} budget. Would that interest you?"
            )
        sys_prompt = "You are the FlowersForYou planner. Protect user privacy and clarify ambiguous requests politely."
        return {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": u_text},
                {"role": "assistant", "content": a_text}
            ]
        }

    def generate_feedback_and_recovery(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        area = rng.choice(CITIES_AREAS[city])
        act = rng.choice(ACTIVITIES["creative"] + ACTIVITIES["culinary"] + ACTIVITIES["social_games"])
        
        issue_types = [
            ("way too loud and packed so we couldn't talk",
             "I'm sorry to hear that the acoustic environment was overwhelming! I've noted a strong preference for quiet, relaxed venues with good table spacing. Next time, would you prefer a peaceful tea lounge with private nooks, or a smaller boutique game parlor that limits group bookings?"),
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
             "Unexpected pricing discrepancies are very frustrating. We only endorse partners who honor verified all-inclusive rates without hidden covers. I can look into verified specialty venues with fixed transparent pricing in this area if you'd like to try again.")
        ]
        issue_desc, recovery_msg = rng.choice(issue_types)
        
        openers = [
            f"The {act} we visited in {area} was {issue_desc}.",
            f"We tried the {act} near {area} yesterday, but it was {issue_desc}.",
            f"Really disappointed with our {act} experience in {area} because it was {issue_desc}.",
            f"Had a frustrating time at the {act} in {area}: it was {issue_desc}."
        ]
        u_text = rng.choice(openers)
        sys_prompt = "You are the FlowersForYou assistant. Handle user feedback with empathy and offer constructive recovery alternatives."
        return {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": u_text},
                {"role": "assistant", "content": recovery_msg}
            ]
        }

    def generate_multi_turn_corrections(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        area1, area2 = rng.sample(CITIES_AREAS[city], 2)
        all_acts = (
            ACTIVITIES["creative"] + ACTIVITIES["culinary"] +
            ACTIVITIES["social_games"] + ACTIVITIES["relaxed_wellness"] +
            ACTIVITIES["active_outdoor"]
        )
        act1, act2 = rng.sample(all_acts, 2)
        b1 = rng.choice([400, 500, 600, 700, 800, 1000])
        b2 = rng.choice([800, 1000, 1200, 1500, 2000])
        timing1 = rng.choice(["Saturday morning", "Saturday afternoon", "Friday evening"])
        timing2 = rng.choice(["Sunday afternoon", "Sunday morning", "next weekend"])
        group = rng.choice([3, 4, 5, 6])
        correction_flavor = rng.choice([
            "activity_and_area", "budget_and_timing", "group_size_expansion",
            "constraint_addition", "vibe_switch"
        ])
        
        sys_prompt = "You are the FlowersForYou assistant. Gracefully handle user corrections and updated preferences without arguing or ignoring changes."

        if correction_flavor == "activity_and_area":
            return {
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"I was thinking about doing a {act1} around {area1} with a budget of ₹{b1}."},
                    {"role": "assistant", "content": f"A {act1} in {area1} sounds wonderful! Are you planning for Saturday or Sunday, and will you be going solo or with friends?"},
                    {"role": "user", "content": f"Actually, change of plans—my friend wants to do a {act2} instead, and our budget is now ₹{b2}. Can we switch to {area2}?"},
                    {"role": "assistant", "content": f"Updated smoothly! I've switched your activity to a {act2}, adjusted the neighborhood to {area2}, and increased your budget to ₹{b2}. What time window works best for you both?"}
                ]
            }
        elif correction_flavor == "budget_and_timing":
            return {
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Can we plan a {act1} in {area1} for {timing1}? Budget is around ₹{b2}."},
                    {"role": "assistant", "content": f"I can help with that! A {act1} around {area1} for {timing1} fits nicely within ₹{b2}. Should I look for morning or afternoon slots?"},
                    {"role": "user", "content": f"Hold on, my budget is actually tighter—keep it under ₹{b1}, and let's move it to {timing2} instead."},
                    {"role": "assistant", "content": f"Got it! Adjusted your budget cap down to ₹{b1} and moved your preferred time to {timing2}. Would you like me to look into options for {timing2} now?"}
                ]
            }
        elif correction_flavor == "group_size_expansion":
            return {
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"I'm looking for a solo spot for a {act1} in {area1} under ₹{b1}."},
                    {"role": "assistant", "content": f"A solo {act1} in {area1} is a great way to unwind. Are you looking for a morning or evening session?"},
                    {"role": "user", "content": f"Actually, {group} of my colleagues want to tag along now! We need a space for {group + 1} people in {area1}."},
                    {"role": "assistant", "content": f"No problem at all! I've updated the plan from a solo outing to a group of {group + 1} in {area1}. Would you like me to find venues that can accommodate group bookings?"}
                ]
            }
        elif correction_flavor == "constraint_addition":
            constraint_text = rng.choice([
                "wheelchair accessibility and step-free entry",
                "strictly vegan culinary options",
                "pet-friendly seating so we can bring a dog",
                "a quiet low-noise environment"
            ])
            return {
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Let's look into a {act1} around {area1} for this weekend."},
                    {"role": "assistant", "content": f"A {act1} in {area1} sounds lovely! What approximate budget are you aiming for?"},
                    {"role": "user", "content": f"Budget is around ₹{b1}. Also, I forgot to mention: one person in our group requires {constraint_text}. Please make sure the venue accommodates that."},
                    {"role": "assistant", "content": f"Important detail noted! I've added a strict filter for {constraint_text} in {area1} with a ₹{b1} budget. Are you looking at Saturday or Sunday?"}
                ]
            }
        else: # vibe_switch
            act_outdoor = rng.choice(ACTIVITIES["active_outdoor"])
            act_relax = rng.choice(ACTIVITIES["relaxed_wellness"])
            return {
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"I was considering an energetic outdoor activity like {act_outdoor} in {area1}."},
                    {"role": "assistant", "content": f"{act_outdoor.title()} in {area1} sounds energizing! Do you have a specific time in mind?"},
                    {"role": "user", "content": f"To be honest, work was exhausting this week. Can we switch to something quiet and relaxing like a {act_relax} instead?"},
                    {"role": "assistant", "content": f"Completely understand—taking it easy is so important. Switched our focus from high energy to a quiet and relaxing {act_relax} in {area1}. Would you prefer Saturday afternoon or Sunday morning?"}
                ]
            }

    def generate_undecided_and_skip(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        area = rng.choice(CITIES_AREAS[city])
        is_skip = rng.random() < 0.5
        
        if is_skip:
            actions = [
                "Bypass the profiling quiz please",
                "I'd prefer not to answer survey questions right now",
                "Skip the onboarding questionnaire",
                "No surveys for me please",
                "Let's skip the profile questions",
                "I want to bypass onboarding",
                "Rather skip the questionnaire",
                "Can we skip the setup questions",
                "Don't feel like doing a survey today",
                "Skip the questions for now",
                "Bypass the intro questions please",
                "I don't want to answer questions",
                "Let's jump over the setup quiz",
                "Skip the questions",
                "No onboarding survey please"
            ]
            transitions = [
                "and jump straight into planning an activity.",
                "let's find something fun to do immediately.",
                "let's pick an outing right away.",
                "just help me pick an activity right now.",
                "let's get right into finding plans.",
                "and jump straight to activity suggestions.",
                "just find me something to do today.",
                "let's dive directly into activities.",
                "let's get straight to recommendations.",
                "let's see what's happening right away."
            ]
            area_context = rng.choice([
                f"for {area}", f"around {area}", f"near {area}", f"in {city}", f"in {area}",
                f"for my weekend in {area}", f"for today in {area}", f"for an outing in {area}"
            ])
            u_text = f"{rng.choice(actions)} {area_context}, {rng.choice(transitions)}"
            a_text = f"Understood—skipping the questions entirely! Let's get straight to planning. Which area around {city} do you want to explore, and what is your approximate budget?"
            sys_prompt = "You are the FlowersForYou onboarding assistant. Respect the user's choice to skip onboarding."
        else:
            indecision_phrases = [
                "I really don't know what I enjoy doing on weekends or what my hobbies are",
                "I'm completely undecided and have trouble figuring out what activity to pick",
                "I have no clue what I'm in the mood for. Everything sounds either too tiring or boring",
                "Not sure what kind of activities fit me best yet",
                "I feel stuck and don't know what kind of outing I want",
                "I honestly have trouble deciding what to do because I'm not sure what my hobbies are",
                "I'm totally torn and don't know what I like yet",
                "My mind is completely blank on what sounds fun this weekend",
                "I'm looking for inspiration because I don't know my activity preferences",
                "I'm new to exploring hobbies and don't know where to start",
                "I can't make up my mind about what to do today",
                "I'm feeling indecisive about what kind of plans to make"
            ]
            anchors = [
                f"for this weekend around {area}.",
                f"for Saturday afternoon near {area}.",
                f"for a solo day off in {city}.",
                f"for a relaxed Sunday in {area}.",
                f"with a friend this weekend.",
                f"for today around {area}."
            ]
            u_text = f"{rng.choice(indecision_phrases)} {rng.choice(anchors)}"
            act1 = rng.choice(ACTIVITIES["creative"])
            act2 = rng.choice(ACTIVITIES["social_games"])
            act3 = rng.choice(ACTIVITIES["relaxed_wellness"])
            a_text = (
                f"That's completely fine—we can discover what you enjoy together! Here are three contrasting options to consider: "
                f"would you prefer a hands-on {act1}, a fun session of {act2}, or a calm {act3}?"
            )
            sys_prompt = "You are the FlowersForYou onboarding assistant. When users are undecided or unsure of their tastes, offer two or three concrete contrasting activity choices rather than open-ended meta-questions."
        
        return {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": u_text},
                {"role": "assistant", "content": a_text}
            ]
        }

    def generate_tool_grounded_search(self, rng: random.Random) -> dict:
        city = rng.choice(list(CITIES_AREAS.keys()))
        area = rng.choice(CITIES_AREAS[city])
        cat = rng.choice(["creative", "culinary", "social_games"])
        act = rng.choice(ACTIVITIES[cat])
        venue_names = [f"Studio {rng.randint(10, 99)}", f"The {act.split()[0].title()} Guild", f"{area} Workshop Space", f"The Urban {act.split()[-1].title()} Spot"]
        venue = rng.choice(venue_names)
        price = rng.choice([650, 750, 850, 950, 1200])
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
                    "amenities": ["materials_provided", "instructor_guided"]
                }
            ]
        }
        u_text = f"Are there any verified {act} sessions available in {area} for this weekend under ₹{price + 200}?"
        tool_text = json.dumps(tool_payload)
        a_text = (
            f"Yes! Based on verified search results, {venue} in {area} offers a {act} at ₹{price} per person with materials provided. "
            f"Current verified open slots are {slot1} and {slot2}. Would you like me to share details on how to reserve one of these slots?"
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


    def generate_candidate_for_category(self, category: str, rng: random.Random) -> dict:
        generators = {
            "first_time_onboarding": self.generate_first_time_onboarding,
            "activity_discovery": self.generate_activity_discovery,
            "structured_preference_extraction": self.generate_structured_preference_extraction,
            "returning_users": self.generate_returning_users,
            "constraints_handling": self.generate_constraints_handling,
            "multi_turn_corrections": self.generate_multi_turn_corrections,
            "undecided_and_skip_onboarding": self.generate_undecided_and_skip,
            "tool_grounded_search": self.generate_tool_grounded_search,
            "booking_safety_and_verification": self.generate_booking_safety_and_verification,
            "ambiguous_and_privacy": self.generate_ambiguous_and_privacy,
            "feedback_and_recovery": self.generate_feedback_and_recovery,
        }
        gen = generators.get(category)
        if not gen:
            raise ValueError(f"Unknown category: {category}")
        return gen(rng)


def generate_batch(
    batch_num: int,
    engine: ScenarioEngine,
    global_seen_prompts: set,
    seed: int = 1000,
) -> tuple[list[dict], list[dict], dict]:
    """Generate exactly 500 accepted examples for a batch."""
    rng = random.Random(seed + batch_num * 101)
    batch_records = []
    manifest_records = []
    rejection_counts = {
        "validation_failure": 0,
        "exact_duplicate": 0,
        "held_out_leakage": 0,
        "near_duplicate": 0,
    }
    batch_id = f"batch_{batch_num:02d}"

    for cat, quota in CATEGORY_BATCH_QUOTAS.items():
        accepted_cat = 0
        attempts = 0
        max_attempts = max(10000, quota * 100)

        while accepted_cat < quota and attempts < max_attempts:
            attempts += 1
            candidate = engine.generate_candidate_for_category(cat, rng)

            # 1. Structural and Safety Validation
            is_valid, errors = validate_single_record(candidate, record_idx=len(batch_records) + 1)
            if not is_valid:
                rejection_counts["validation_failure"] += 1
                continue

            # 2. Duplicate Check
            u_text = get_conversation_user_text(candidate.get("messages", []))
            u_norm = " ".join(u_text.lower().split())
            if u_norm in global_seen_prompts:
                rejection_counts["exact_duplicate"] += 1
                continue

            # 3. Held-Out Leakage Check
            if engine._check_leakage(u_text):
                rejection_counts["held_out_leakage"] += 1
                continue

            # Accept candidate
            global_seen_prompts.add(u_norm)
            accepted_cat += 1
            ex_id = f"ffy_syn_{batch_num:02d}_{len(batch_records) + 1:03d}"
            batch_records.append(candidate)

            # Build provenance manifest record
            manifest_records.append({
                "id": ex_id,
                "source_type": "synthetic",
                "category": cat,
                "batch_id": batch_id,
                "consent_status": "not_applicable_synthetic",
                "dialogue_turns": len(candidate.get("messages", [])),
                "has_tool": any(m.get("role") == "tool" for m in candidate.get("messages", [])),
            })

        if accepted_cat < quota:
            raise RuntimeError(
                f"Category '{cat}' in {batch_id} only generated {accepted_cat}/{quota} after {attempts} attempts!"
            )

    return batch_records, manifest_records, rejection_counts


def write_batch_report(
    batch_id: str,
    batch_records: list[dict],
    manifest_records: list[dict],
    rejections: dict,
    audit_res: dict,
    out_path: Path,
):
    cat_counts = {}
    for m in manifest_records:
        cat_counts[m["category"]] = cat_counts.get(m["category"], 0) + 1

    content = f"""# FlowersForYou Batch Report: {batch_id.upper()}

- **Batch ID:** `{batch_id}`
- **Accepted Records:** {len(batch_records)}
- **Validation Status:** {"PASS" if audit_res['passed'] else "FAIL"}
- **Max Eval Jaccard:** {audit_res['max_eval_jaccard']:.3f}
- **Held-Out Leakages:** {len(audit_res['held_out_leakages'])}
- **Exact Duplicates:** {audit_res['exact_duplicates']}

## Category Distribution
| Category | Accepted Count | Target |
|---|---:|---:|
"""
    for cat, quota in CATEGORY_BATCH_QUOTAS.items():
        actual = cat_counts.get(cat, 0)
        content += f"| `{cat}` | {actual} | {quota} |\n"

    content += f"""
## Rejection & Filtering Metrics
- **Validation / Safety Rejections:** {rejections['validation_failure']}
- **Exact Duplicate Rejections:** {rejections['exact_duplicate']}
- **Held-Out Leakage Rejections:** {rejections['held_out_leakage']}
- **Near Duplicate Rejections:** {rejections['near_duplicate']}

## Quality Checks
- Structural errors: {len(audit_res['structural_errors'])}
- Safety violations: {len(audit_res['safety_violations'])}
- Manifest synchronization: {"Verified" if audit_res['manifest_sync'] else "Desynchronized"}
"""
    out_path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate 10,000 FlowersForYou dataset.")
    parser.add_argument("--num-batches", type=int, default=20, help="Number of 500-example batches (default: 20).")
    parser.add_argument("--batch", type=int, default=None, help="Generate single batch index (1-20).")
    args = parser.parse_args()

    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    held_out = load_held_out_eval()
    engine = ScenarioEngine(held_out)
    global_seen_prompts = set()

    print("=" * 70)
    print("FlowersForYou 10,000-Example Dataset Generator")
    print("=" * 70)
    print(f"Target Batches:       {args.num_batches}")
    print(f"Examples per Batch:   500")
    print(f"Total Target Records: {args.num_batches * 500}")
    print(f"Held-Out Eval Set:    {len(held_out)} records")
    print("=" * 70)

    all_accepted_records = []
    all_manifest_records = []
    total_rejections = {"validation_failure": 0, "exact_duplicate": 0, "held_out_leakage": 0, "near_duplicate": 0}

    batches_to_run = [args.batch] if args.batch else list(range(1, args.num_batches + 1))

    for b_idx in batches_to_run:
        b_id = f"batch_{b_idx:02d}"
        b_file = BATCHES_DIR / f"{b_id}.jsonl"
        m_file = BATCHES_DIR / f"{b_id}_manifest.jsonl"
        r_file = BATCH_REPORTS_DIR / f"{b_id}_report.md"

        print(f"\n--- Generating {b_id.upper()} (Target: 500 records) ---")
        recs, manifests, rejections = generate_batch(b_idx, engine, global_seen_prompts, seed=2026 + b_idx * 50)
        for k in total_rejections:
            total_rejections[k] += rejections[k]

        # Save immutable batch
        with open(b_file, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        with open(m_file, "w", encoding="utf-8") as f:
            for m in manifests:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        # Run 10-point audit on batch
        audit_res = audit_batch_or_dataset(b_file, manifest_path=m_file, target_count=500)
        write_batch_report(b_id, recs, manifests, rejections, audit_res, r_file)

        status_badge = "[PASS]" if audit_res["passed"] else "[FAIL]"
        print(f"{status_badge} {b_id}: {len(recs)} records written | Max Eval Jaccard: {audit_res['max_eval_jaccard']:.3f}")
        if not audit_res["passed"]:
            print(f"[ERROR] Batch {b_id} failed audit!")
            sys.exit(1)

        all_accepted_records.extend(recs)
        all_manifest_records.extend(manifests)

    # When all 20 batches are complete, compile the full 10,000 dataset
    if len(batches_to_run) == args.num_batches:
        print("\n" + "=" * 70)
        print("Compiling Full Expanded Dataset & Sidecar Manifest...")
        print("=" * 70)

        with open(EXPANDED_TRAIN_FILE, "w", encoding="utf-8") as f:
            for r in all_accepted_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            for m in all_manifest_records:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        print(f"Compiled: {EXPANDED_TRAIN_FILE} ({len(all_accepted_records)} records)")
        print(f"Manifest: {MANIFEST_FILE} ({len(all_manifest_records)} records)")

        # Run comprehensive audit on final dataset
        full_audit = audit_batch_or_dataset(
            EXPANDED_TRAIN_FILE,
            manifest_path=MANIFEST_FILE,
            target_count=10000,
        )
        print("\nFull Dataset Audit Summary:")
        print(f"  - Total Records:     {full_audit['total_records']}")
        print(f"  - Valid Records:     {full_audit['valid_records']}")
        print(f"  - Safety Violations: {len(full_audit['safety_violations'])}")
        print(f"  - Exact Duplicates:  {full_audit['exact_duplicates']}")
        print(f"  - Max Eval Jaccard:  {full_audit['max_eval_jaccard']:.3f}")
        print(f"  - Held-Out Leakage:  {len(full_audit['held_out_leakages'])}")
        print(f"  - Manifest Sync:     {full_audit['manifest_sync']}")
        print(f"  - Overall Pass:      {full_audit['passed']}")


if __name__ == "__main__":
    main()
