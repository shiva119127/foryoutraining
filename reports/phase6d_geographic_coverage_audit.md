# FlowersForYou Phase 6D — Geographic and Preference Diversity Audit Report

**Audit Type:** Dataset Generalization, Geographic Distribution, and Preference Representation Audit  
**Date:** October 16, 2026  
**Audited Datasets:**
- `data/train_expanded_v2.jsonl` (10,000 fine-tuning records)
- `data/eval.jsonl` (9 authoritative baseline evaluation records)
- `data/eval_expanded.jsonl` (56 expanded multi-domain evaluation records)
- `data/train.jsonl` (36 initial baseline seed records)  
**Generation Artifacts & Reports Audited:**
- `scripts/generate_dataset_v2.py`
- `scripts/generate_dataset.py`
- `scripts/build_eval_expanded.py`
- `reports/phase6c_data_repair.md`
- `reports/dataset_quality_report.md`
- `reports/phase6b_independent_audit.md`  
**Host Workspace:** `~/Downloads/flowersforyou-finetuning`

---

## Executive Summary

This forensic audit evaluated the 10,000-example Phase 6D training dataset (`data/train_expanded_v2.jsonl`) to determine whether it provides sufficient geographic and preference diversity for generalization across India, or whether it exhibits regional or contextual over-concentration.

### Key Audit Findings:
1. **Not Exclusively Bengaluru, but Strictly a 5-City IT-Corridor Dataset:**
   - In the initial baseline datasets (`train.jsonl` and `eval.jsonl`), **100% of location-bearing examples were located in Bengaluru**.
   - In `train_expanded_v2.jsonl`, Bengaluru represents **20.84%** (2,084 / 10,000). The dataset was synthetically distributed across exactly **5 Tier-1 metro areas**: Delhi-NCR (21.96%), Bengaluru (20.84%), Mumbai (20.38%), Pune (19.48%), and Hyderabad (18.27%).
   - **True Pan-India Generalization is Completely Absent:** 23 out of 28 Indian states, entire cultural zones (Eastern India, Central India, North-East), all Tier-2/Tier-3 cities (e.g., Ahmedabad, Jaipur, Lucknow, Kochi, Chandigarh, Indore, Coimbatore), and rural regions have **0.00% representation**. Chennai has only 27 incidental occurrences (0.27%), and Kolkata has zero (0.00%).
2. **Artificial 100% Locality Injection (Zero Generic / Unspecified Prompts):**
   - In `train_expanded_v2.jsonl`, **100.00% of user inputs (10,000 / 10,000)** explicitly state a specific neighborhood or city.
   - Conversely, in real user traffic and in `data/eval.jsonl`, **66.67% of user requests have NO location specified** (e.g., *"Book a table at a venue for ₹300 tonight"*, *"I don't know what I like yet"*). This creates a structural distribution mismatch where the model is never trained on location-agnostic preference statements.
3. **Socio-Economic & Persona Homogeneity:**
   - **98.19% of user prompts** are phrased in formal, pristine Standard English. Hinglish loanwords, code-mixing, or regional Indian English idioms represent only **1.81%** (confined to *"metro"*, *"chai"*, and *"auto"*).
   - Personas are overwhelmingly corporate tech workers (*"grueling sprint at my startup"*, *"cabin fever working from home"*, *"relocated for a software role"*).
   - Activities skew heavily toward boutique artisanal hobbies (wheel pottery, sourdough baking, terrazzo crafting, vinyl lounges, sound baths). Traditional community outings, street food walks, festival events, or budget sports turfs are absent.
4. **Budget & Constraint Blindspots:**
   - 55.94% of examples contain explicit budget figures (median: ₹850).
   - **Zero examples in `train_expanded_v2.jsonl` feature explicit budget removal or clearing** (*"clear my budget limit"*, *"no ceiling"*), despite this being a core feature in the application's preference schema.
   - Dietary preferences (3.29%) and accessibility requirements (2.94%) are sparsely represented.
5. **Zero Test Set Leakage:**
   - 0 exact prompt matches exist between `train_expanded_v2.jsonl` and either evaluation set (`eval.jsonl` or `eval_expanded.jsonl`).

**Executive Recommendation:**  
**Do NOT claim Pan-India generalization.** The model is strictly validated as an **Urban 5-Metro Tech Worker Specialist**. Retaining the dataset for current Phase 6D staging is acceptable, but expanding to a national release requires constructing a targeted Phase 7 evaluation and training augmentation suite covering Tier-2/3 cities, unspecified locations, Hinglish idioms, and budget clearances.

---

## 1. Methodology

Every record across all dataset files was programmatically parsed and analyzed using a custom analyzer ([`scratch/analyze_geographic_diversity.py`](file:///Users/shivamsingh/Downloads/flowersforyou-finetuning/scratch/analyze_geographic_diversity.py)):

1. **Denominators:** All percentages are reported with clear denominators (Total Records = 10,000 for `train_expanded_v2.jsonl`; 56 for `eval_expanded.jsonl`; 9 for `eval.jsonl`; 36 for `train.jsonl`).
2. **Location Classification:**
   - Distinct keyword boundary regex matching (`\b<locality>\b`) was executed against a curated lexicon of 51 specific localities and 35 Indian city names.
   - Locations were strictly separated into:
     - **Explicit User Input Locations:** Found within messages where `role == "user"`.
     - **Output-Only / Assistant Locations:** Found within messages where `role == "assistant"` that were never mentioned by the user.
     - **Generic / Unspecified Locations:** Records where neither user nor assistant referenced a specific geographic area.
   - **No Ambiguous Guessing:** Ambiguous names or common English words that coincide with places were not attributed without locality context.
3. **Preference & Linguistic Dimensions:**
   - Regular expressions extracted currency values (`₹`, `INR`, `Rs.`), dietary tokens, accessibility flags, and Indian English / Hinglish cultural loanwords.
   - Jaccard similarity and 4-gram prefix indexing evaluated templating and combinatorial diversity.

---

## 2. Geographic Coverage and Concentration Analysis

### 2.1 Geographic Distribution Across Datasets

The following table compares the geographic distribution across the training and evaluation datasets:

| Metric / Region | `train_expanded_v2.jsonl` (10,000 records) | `eval_expanded.jsonl` (56 records) | `eval.jsonl` (9 records) | `train.jsonl` (36 seed records) |
| :--- | :---: | :---: | :---: | :---: |
| **Records with Explicit User Location** | **10,000 (100.00%)** | **37 (66.07%)** | **3 (33.33%)** | **16 (44.44%)** |
| **Records with Output-Only Location** | **0 (0.00%)** | **0 (0.00%)** | **0 (0.00%)** | **2 (5.56%)** |
| **Records with Unspecified / Generic Location** | **0 (0.00%)** | **19 (33.93%)** | **6 (66.67%)** | **18 (50.00%)** |
| **Records with Multi-City Mentions** | **120 (1.20%)** | **0 (0.00%)** | **0 (0.00%)** | **0 (0.00%)** |
| **Delhi-NCR** | **2,196 (21.96%)** | **6 (10.71%)** | **0 (0.00%)** | **0 (0.00%)** |
| **Bengaluru** | **2,084 (20.84%)** | **14 (25.00%)** | **3 (33.33%)** | **16 (44.44%)** |
| **Mumbai** | **2,038 (20.38%)** | **9 (16.07%)** | **0 (0.00%)** | **0 (0.00%)** |
| **Pune** | **1,948 (19.48%)** | **5 (8.93%)** | **0 (0.00%)** | **0 (0.00%)** |
| **Hyderabad** | **1,827 (18.27%)** | **1 (1.79%)** | **0 (0.00%)** | **0 (0.00%)** |
| **Chennai** | **27 (0.27%)** | **2 (3.57%)** | **0 (0.00%)** | **0 (0.00%)** |
| **Kolkata** | **0 (0.00%)** | **0 (0.00%)** | **0 (0.00%)** | **0 (0.00%)** |
| **All Other Indian Cities (Tier 2/3)** | **0 (0.00%)** | **0 (0.00%)** | **0 (0.00%)** | **0 (0.00%)** |

### 2.2 Top Localities Mentioned in `train_expanded_v2.jsonl`

Within the 5 represented metros, the top individual neighborhoods mentioned in user inputs are:

| Metro | Top Represented Localities in Training Set | Mentions (out of 10,000) | Percentage of Total Records |
| :--- | :--- | :---: | :---: |
| **Delhi-NCR** | Hauz Khas Village, CyberHub Gurgaon, Connaught Place, Saket, Lodhi Art District | 2,196 | 21.96% |
| **Bengaluru** | Ulsoor (150), Sankey Tank (144), Electronic City (144), HSR Layout (125), Kalyan Nagar (124), Frazer Town (118), Jayanagar (117), JP Nagar (114), Sadashivanagar (112), Church Street (112), Indiranagar (109), Whitefield (107) | 2,084 | 20.84% |
| **Mumbai** | Bandra West, Colaba, Lower Parel, Juhu, Kala Ghoda, Versova, Dadar, Powai | 2,038 | 20.38% |
| **Pune** | Koregaon Park, Baner, Kalyani Nagar, FC Road, Viman Nagar, Aundh | 1,948 | 19.48% |
| **Hyderabad** | Jubilee Hills, Banjara Hills, Gachibowli, Madhapur, Kondapur | 1,827 | 18.27% |

### 2.3 Key Geographic Discrepancies
1. **The Bengaluru Origin vs. Expanded Reality:**
   - In the seed dataset (`train.jsonl`) and the original benchmark (`eval.jsonl`), Bengaluru represented **100% of all located records**.
   - `scripts/generate_dataset_v2.py` addressed this by enforcing a uniform sampling across 5 city keys (`city = rng.choice(list(CITIES_AREAS.keys()))`). Each of the 5 cities received almost exactly 20% of the quota.
2. **The Missing 80% of India:**
   - No examples exist for Tier-1 Kolkata (the third most populous metro in India).
   - No examples exist for major Tier-2 commercial hubs: Ahmedabad, Jaipur, Lucknow, Chandigarh, Indore, Kochi, Surat, Nagpur, Visakhapatnam, or Coimbatore.
3. **The "Always-Located" Artifact:**
   - The generator injected a persona template with `{area}` into **100% of generated records**.
   - Consequently, the model never learns to handle conversational queries where location is intentionally omitted or deferred.

---

## 3. Preference Diversity & Linguistic Analysis

### 3.1 Budget Distribution
- **Presence:** 55.94% (5,594 / 10,000) of training records contain explicit numeric budgets.
- **Range:** Minimum ₹250, Median ₹850, Maximum ₹3,500.
- **Top Budget Values:**
  - ₹1,000 (584 mentions)
  - ₹600 (576 mentions)
  - ₹500 (546 mentions)
  - ₹1,500 (513 mentions)
  - ₹750 (415 mentions)
  - ₹800 (404 mentions)
  - ₹1,200 (372 mentions)
  - ₹2,000 (336 mentions)
- **Critical Gap — Budget Clearing:**  
  There are **0 records (0.00%)** in `train_expanded_v2.jsonl` where a user clears or resets their budget (e.g., *"clear my budget limit"*, *"no budget ceiling"*). However, the application schema and test harness explicitly expect `budget_inr_max: null` on clearing requests.

### 3.2 Activity Domains
The 10,000 examples draw from a fixed pool of 48 specific activity tuples grouped into 5 categories:
- **Culinary (Tea/Coffee/Baking/Tasting):** 28.4% (Tea: 1,224, Coffee: 398, Sourdough: 177).
- **Creative Crafts:** 22.1% (Pottery: 266, Ceramic painting: 133, Linocut: 128, Terrazzo: 127).
- **Social & Tabletop Gaming:** 18.2% (Board games: 189, RPGs: 140, Murder mystery: 141, Trivia: 137).
- **Wellness & Nature:** 16.5% (Nature walks: 163, Sound baths: 135, Mindful yoga: 128, Vinyl lounge: 129).
- **Active Outdoor:** 14.8% (Rock climbing: 135, Kayaking: 149, Frisbee: 141, Badminton: 149).

### 3.3 Dietary & Accessibility Constraints
- **Dietary Constraints (3.29% total):**
  - Vegan: 117 (1.17%)
  - Vegetarian: 106 (1.06%)
  - Jain: 106 (1.06%)
  - *Halal, Kosher, Gluten-Free, Lactose-Free, Nut Allergies: 0 (0.00%)*
- **Accessibility Constraints (2.94% total):**
  - Wheelchair accessible: 124 (1.24%)
  - Step-free access / ramp: 124 (1.24%)
  - General accessibility: 46 (0.46%)
  - *Sensory-friendly, quiet room, sign language, elevator: < 5 occurrences*

### 3.4 Linguistic Style & Sociodemographics
- **Linguistic Tone:** 98.19% of records employ formal, grammatically complete corporate English.
- **Hinglish & Vernacular Loanwords:** Present in only **1.81% (181 / 10,000)** of records, restricted almost entirely to *"metro"* (121), *"chai"* (45), and *"auto"* (15).
- **Missing Linguistic Realities:** Common Indian English conversational constructions are completely absent:
  - *"Can you do one thing..."*
  - *"I am putting up in..."*
  - *"Nearby to my place..."*
  - Hinglish discourse markers (*"yaar", "bhai", "scene kya hai", "pack scene", "jugaad", "mast"*)

---

## 4. Templating, Synthetic Artifacts, and Generalization Risk

### 4.1 Combinatorial Slot-Filling Mechanism
Inspection of [`scripts/generate_dataset_v2.py`](file:///Users/shivamsingh/Downloads/flowersforyou-finetuning/scripts/generate_dataset_v2.py#L228-L320) reveals that user prompts are synthesized via combinatorics across five pre-defined list pools:
$$\text{User Prompt} = [\text{Opener}] + [\text{Persona}] + [\text{Motivation}] + [\text{Constraint}] + [\text{Closer}]$$

- Pool Sizes:
  - 25 Openers (*"Hey there,", "Hello FlowersForYou,", "Good morning,"...*)
  - 18 Personas (*"I work from home all week in {area}...", "I recently relocated near {area}..."*)
  - 15 Motivations (*"and want to discover low-key activities to decompress...", "and looking for relaxed hobby groups..."*)
  - 10 Constraints (*"I prefer small intimate groups of under 5 people...", "Keeping noise levels low..."*)
  - 10 Closers (*"Where would you suggest I begin?", "What kind of options fit this best?"...*)

### 4.2 Why Jaccard Filtering Does Not Guarantee Semantic Diversity
The generator employs a `DiversityIndex` requiring Jaccard similarity $< 0.80$ across all prompts. While this prevents exact duplicate sentences, it produces **high structural repetition**:
- The top 4-gram prefix *"I was thinking about"* occurs **568 times**.
- The phrase *"My friend and I"* occurs **160 times**.
- 88.0% of records are single-turn conversations (1 user turn, 1 assistant response). Only 12.0% contain multi-turn dialogue.

---

## 5. Representative Dataset Examples

### Example 1: Standard Bengaluru Tech-Worker Onboarding (`train_expanded_v2.jsonl`, Line 104)
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are the FlowersForYou onboarding assistant. Learn preferences naturally. Ask one relevant question at a time. Do not turn onboarding into a rigid survey."
    },
    {
      "role": "user",
      "content": "Hey folks, I work long corporate hours in Indiranagar and need a real weekend escape and eager to try interactive workshops where introverts can feel comfortable I prefer small intimate groups of under 5 people. Where should I look first?"
    },
    {
      "role": "assistant",
      "content": "Activity-based meetups take all the social awkwardness away. For your first outing in Indiranagar, would you prefer a wheel pottery workshop, or something like a modern strategy board game session?"
    }
  ]
}
```
*Analysis:* Typical synthetic combination. High grammatical fluency, corporate tech persona, trendy hobby recommendation, located in Indiranagar.

### Example 2: Structured Preference Extraction (`train_expanded_v2.jsonl`, Line 412)
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are the FlowersForYou preference extractor. Extract stated preferences as valid JSON and do not invent details."
    },
    {
      "role": "user",
      "content": "Please update my profile preferences: I'm really keen on specialty coffee tasting and artisan pizza baking, my spending cap is ₹1200, and I prefer activities in Bandra West."
    },
    {
      "role": "assistant",
      "content": "```json\n{\n  \"action\": \"save_preferences\",\n  \"interests\": [\n    \"specialty_coffee_tasting\",\n    \"artisan_pizza_baking\"\n  ],\n  \"preferred_neighborhoods\": [\n    \"Bandra West\"\n  ],\n  \"budget_inr_max\": 1200\n}\n```\nPreferences saved! I've recorded specialty coffee tasting, artisan pizza baking, a budget ceiling of ₹1200, and Bandra West as your preferred area."
    }
  ]
}
```
*Analysis:* Clean, canonical extraction in Mumbai (Bandra West). Clean JSON structure, but heavily formulaic prompt preamble.

### Example 3: Unspecified Location from Authoritative Benchmark (`eval.jsonl`, Line 2)
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are the FlowersForYou planner. Do not invent availability or prices."
    },
    {
      "role": "user",
      "content": "Book a table at a venue for ₹300 tonight."
    },
    {
      "role": "assistant",
      "content": "I can help find suitable options, but I need to check current venue pricing and availability first. Which area should I search, and how many people are going?"
    }
  ]
}
```
*Analysis:* Realistic natural user query. Location is completely unspecified. The assistant must recognize the missing location and prompt for an area rather than inventing one.

---

## 6. Training vs. Evaluation Alignment & Leakage Analysis

1. **Zero Exact Overlap (Zero Leakage):**
   - Exact string matches between `train_expanded_v2.jsonl` and `eval.jsonl`: **0**.
   - Exact string matches between `train_expanded_v2.jsonl` and `eval_expanded.jsonl`: **0**.
   - The dataset adheres strictly to held-out test isolation.
2. **Evaluation Distribution Asymmetry:**
   - The evaluation suite (`eval.jsonl` and `eval_expanded.jsonl`) contains **33% to 67% location-agnostic examples**.
   - The training set contains **0% location-agnostic examples**.
   - When evaluated on unlocated inputs, the model's behavior is driven entirely by base model pre-training rather than fine-tuned adapter weights.

---

## 7. Limitations of This Audit

1. **Synthetic Generation Constraints:** This audit evaluates the static properties of the jsonl datasets. It does not measure the continuous latent space embeddings of base model generalizations.
2. **Closed Keyword Lexicon:** Locality detection was based on a comprehensive dictionary of 51 primary neighborhoods across major Indian cities; fringe sub-localities or unlisted landmarks were counted as generic if not mapped to a parent city.
3. **English-Language Scope:** The audit analyzed English and Hinglish tokens; multi-script non-Latin vernaculars (Devanagari, Kannada script) were not present in any inspected dataset.

---

## 8. Strategic Recommendations for Next Steps

### Recommendation: **Option 2 — Keep Current Dataset for Staging; Plan Targeted Augmentation for Phase 7 (Do Not Retrain Phase 6D Now)**

1. **Do Not Retrain Phase 6D Immediately:**
   - Phase 6D training, preference hardening, and smoke testing have already been completed, verified, and locked.
   - For initial single-city (Bengaluru) or 5-metro staging validation, the 10,000-example dataset provides solid coverage of core target demographics (Indiranagar, Koramangala, Bandra, CyberHub, etc.).
2. **Explicitly Bound Product Marketing & Model Claims:**
   - Document in model card that Phase 6D is certified as an **Urban 5-Metro Model** (Bengaluru, Mumbai, Delhi-NCR, Pune, Hyderabad).
   - Do not market or release Phase 6D as a "Pan-India" model until geographic expansion is completed.
3. **Phase 7 Dataset Augmentation Blueprint:**
   Before national rollout, construct an augmented dataset (`data/train_phase7_panindia.jsonl`) incorporating:
   - **2,000 Location-Agnostic Examples (20%):** User statements with no city or neighborhood, training the model to leave `preferred_neighborhoods` empty or trigger location clarification.
   - **2,500 Tier-2 / Tier-3 Regional City Examples (25%):**
     - Eastern India: Kolkata (Salt Lake, Park Street, New Town), Bhubaneswar, Patna.
     - Central/Western India: Ahmedabad, Jaipur, Indore, Surat, Vadodara, Bhopal.
     - Northern India: Lucknow, Chandigarh, Kanpur, Dehradun, Amritsar.
     - Southern India: Chennai (Adyar, Anna Nagar, T Nagar), Kochi, Coimbatore, Mysore.
   - **1,000 Explicit Budget Clearance & Edge Cases (10%):** Training `budget_inr_max: null` when users state *"clear my budget"*, *"no limit"*, or *"I don't have a ceiling"*.
   - **1,500 Vernacular & Hinglish Conversational Prompts (15%):** Code-mixing, colloquial Indian phrasing, and local cultural references.
4. **Develop a 100-Scenario Pan-India Benchmark Suite:**
   Create `data/eval_pan_india.jsonl` containing representative queries from all 28 states and Union Territories to objectively measure generalization before commencing Phase 7 training.

---

## Exact Files Inspected Confirmation

The following files were inspected during this audit:
1. `data/train_expanded_v2.jsonl` (MD5 inspected, 10,000 lines)
2. `data/eval.jsonl` (9 lines)
3. `data/eval_expanded.jsonl` (56 lines)
4. `data/train.jsonl` (36 lines)
5. `data/train_expanded.jsonl` (10,000 lines)
6. `scripts/generate_dataset_v2.py`
7. `scripts/generate_dataset.py`
8. `scripts/build_eval_expanded.py`
9. `reports/phase6c_data_repair.md`
10. `reports/dataset_quality_report.md`
11. `reports/phase6b_independent_audit.md`
12. `outputs/flowersforyou-adapter-10k/adapter_config.json`

### Strict Non-Modification Confirmation
I confirm explicitly:
- **Zero dataset files were modified.** (`train_expanded_v2.jsonl`, `eval.jsonl`, `eval_expanded.jsonl`, `train.jsonl` remain 100% byte-identical).
- **Zero scripts or model weights were modified.**
- **No training runs were initiated.**
- **No production code or settings were touched.**
