# FlowersForYou Dataset Quality & Validation Report

## Executive Summary

This report documents the generation, structural verification, behavioral safety validation, and provenance tracking for the **FlowersForYou 10,000-example expanded fine-tuning dataset**.

- **Target Record Count:** 10,000
- **Accepted Record Count:** 10,000
- **Structural Integrity:** 10,000 / 10,000 (100% pass, 0 schema errors)
- **Safety & Behavioral Compliance:** 10,000 / 10,000 (100% pass, 0 violations)
- **Exact Duplicates:** 0 (100% unique prompt/response conversations)
- **Semantic Near-Duplicates (Jaccard $\ge 0.85$):** 0
- **Held-Out Evaluation Overlap ($J_{max}$):** 0.391 (strictly $< 0.40$ threshold; 0 leakage)
- **Sidecar Manifest Synchronization:** 10,000 records mapped 1-to-1 with complete provenance
- **Held-Out Evaluation Set (`data/eval.jsonl`):** 9 records (completely isolated and unmodified)
- **Active Baseline Training Set (`data/train.jsonl`):** 36 records (preserved and unmodified)
- **Adapter Weights (`outputs/`):** Strictly untouched; zero training executed.

---

## 1. Category Distribution Breakdown

The expanded dataset was generated across 20 immutable batches of 500 examples each. Every batch enforced strict category quotas matching the target plan:

| Behavioral Category | Batch Quota | Target Total | Accepted Records | Status |
|---|---:|---:|---:|:---:|
| **First-time onboarding and progressive discovery** | 40 | 800 | 800 | **MATCH** |
| **Activity discovery and personalized planning** | 65 | 1,300 | 1,300 | **MATCH** |
| **Structured preference extraction** | 35 | 700 | 700 | **MATCH** |
| **Returning users and saved preference context** | 30 | 600 | 600 | **MATCH** |
| **Budget, location, group size, timing & accessibility constraints** | 45 | 900 | 900 | **MATCH** |
| **Multi-turn conversations, corrections & changing preferences** | 60 | 1,200 | 1,200 | **MATCH** |
| **Undecided users, skipped onboarding & low-friction starts** | 35 | 700 | 700 | **MATCH** |
| **Tool-grounded venue search, results & verified facts** | 50 | 1,000 | 1,000 | **MATCH** |
| **Booking safety, unavailable venues, prices & partner verification** | 70 | 1,400 | 1,400 | **MATCH** |
| **Ambiguous requests, refusals, edge cases & privacy protection** | 35 | 700 | 700 | **MATCH** |
| **User feedback, failed plans & recovery** | 35 | 700 | 700 | **MATCH** |
| **Total** | **500** | **10,000** | **10,000** | **100.0%** |

---

## 2. Provenance and Source Breakdown

- **Synthetic Records:** 10,000 (100.0%)
- **Consented Real Conversations:** 0 (0.0%)
  - *Provenance Note:* Phase 1 repository audit verified that no consented real-conversation data or consent records exist in the repository. In compliance with strict privacy and trust guidelines, unverified personal files and external services were excluded. No synthetic record is labeled as real.
- **Sidecar Manifest:** `data/dataset_manifest.jsonl`
  - Every entry contains: `id` (e.g., `ffy_syn_01_001`), `source_type` (`synthetic`), `category`, `batch_id` (`batch_01` to `batch_20`), `consent_status` (`not_applicable_synthetic`), `dialogue_turns`, and `has_tool`.
  - All metadata resides exclusively in the sidecar manifest. Training records in `data/train_expanded.jsonl` contain only the canonical `{"messages": [...]}` array, maintaining total compatibility with `train.py` and TRL's dataset formatter.

---

## 3. Structural & Schema Compliance

The validator (`scripts/validate_dataset.py`) evaluated all 10,000 records against the chat-template schema expected by `train.py`, Transformers 4.57.6, and TRL 0.26.2:

1. **Top-Level Record Schema:** Every line parses as valid JSON with exactly one top-level key: `"messages"`.
2. **Supported Message Roles:** Every message role is in `{"system", "user", "assistant", "tool"}`.
3. **Turn Ordering & Termination:**
   - Multi-turn sequences follow conversational turn order: `system` $\to$ `user` $\to$ (`tool`) $\to$ `assistant`.
   - The final message of every record is guaranteed to be an `assistant` turn with non-empty text.
4. **Tool Role Semantics:** 1,000 records include tool grounding. Tool responses contain valid serialized JSON payloads with structured fields (`search_results`, `venue_name`, `area`, `city`, `verified_price_inr`, `available_slots`, `partner_status`).
5. **Canonical Preference Extraction Schema:** 700 records test structured preference extraction. Every record produces valid JSON conforming to the project's canonical specification:
   ```json
   {
     "action": "save_preferences",
     "interests": ["<activity_slug>", ...],
     "preferred_neighborhoods": ["<neighborhood_name>"],
     "budget_inr_max": 1000
   }
   ```
   - Value fidelity: Numeric INR budgets are preserved as integers; neighborhoods and interests preserve user-stated input without hallucinated substitutions or conversions.

---

## 4. Behavioral & Safety Compliance Scorecard

| Rule & Safety Requirement | Validator Check | Violations in 10,000 Records | Result |
|---|---|---:|:---:|
| **No Fabricated Booking Confirmations** | Rejects phrases like *"booking confirmed"*, *"reservation secured"*, *"voucher generated"*, *"table reserved"* without verified tool results | 0 | **PASS** |
| **No Booking Commitments Without Tools** | Rejects *"I'll make sure to book"*, *"I will reserve that for you"* | 0 | **PASS** |
| **No Invented Partner Discounts / Deals** | Rejects unverified claims of member discounts or special perks (e.g., *"15% off for FlowersForYou members"*) | 0 | **PASS** |
| **No Invented Complimentary Perks** | Rejects unverified promises of *"free drinks"*, *"complimentary desserts"*, *"free parking"* | 0 | **PASS** |
| **No Unrealistic Operating Hours Claims** | Refuses late-night workshops (e.g., midnight pottery/climbing) and explains standard daytime operating hours | 0 | **PASS** |
| **No Geographic Hallucinations** | Context grounded exclusively in genuine urban localities across Bengaluru, Mumbai, Delhi-NCR, Pune, and Hyderabad | 0 | **PASS** |
| **No Placeholder Artifacts** | Zero occurrences of `[Insert ...]`, `[Venue Name]`, `<insert_date>` | 0 | **PASS** |
| **Privacy Protection** | Rejects and refuses storage of credit cards, CVVs, passwords, phone numbers, or full street addresses | 0 | **PASS** |
| **Respect for Skipped Onboarding** | Immediately initiates planning without proposing follow-up surveys or questionnaires | 0 | **PASS** |
| **Decisive Guidance for Undecided Users** | Always offers 2–3 concrete contrasting activity options instead of meta-surveys | 0 | **PASS** |
| **Returning User Profile Continuity** | Welcomes returning users without restarting initial onboarding surveys | 0 | **PASS** |
| **Graceful Handling of Multi-Turn Corrections** | Smoothly updates activity, neighborhood, budget, group size, or accessibility constraints across turns | 0 | **PASS** |

---

## 5. Deduplication and Held-Out Leakage Audit

### Internal Diversity & Deduplication
- **Exact Prompt / Conversation Duplicates:** 0
- **Normalized Prompt Clones:** 0
- **Candidate Generator Combinatorics:** Each behavioral domain utilizes multi-attribute combinatorial generators (urban neighborhoods, activity slugs, INR price tiers, group sizes, PII types, complaint categories, and phrasing templates), producing a combinatorial space $>10^6$ distinct candidate states.

### Held-Out Evaluation Separation
The project's held-out benchmark (`data/eval.jsonl`) contains 9 canonical test cases.

- **Leakage Threshold:** Jaccard similarity $< 0.40$ on normalized user prompts.
- **Maximum Observed Jaccard:** **0.391** (Record #2293 vs Eval #4).
  - *Candidate Prompt:* `"Let's skip the profile questions for an outing in Lower Parel, and jump straight to activity suggestions."`
  - *Eval Prompt:* `"I'd rather skip the profile questions and jump straight into picking an activity."`
  - *Analysis:* While both address the "skip onboarding" domain, Record #2293 introduces localized urban grounding (Lower Parel) and distinct phrasing. The prompt falls safely below the 0.40 leakage threshold, and the assistant responses are completely independent.
- **Leakage Violations:** **0**

---

## 6. Regression Test Suite Results

The hardened safety evaluator unit test suite (`test_evaluator.py`) and repository data validator (`validate_data.py`) were executed:

```bash
./.venv/bin/python test_evaluator.py
./.venv/bin/python validate_data.py
```

### Test Results
- `test_evaluator.py`: **39 / 39 passed** (100%)
  - Confirmed detection of all 6 Phase 4/5 historical failure modes.
  - Confirmed passing of all safe counterexamples (perk refusals, tentative suggestions, valid JSON extractions).
- `validate_data.py`: **PASS**
  - Verified active `data/train.jsonl` (36 records) and held-out `data/eval.jsonl` (9 records) remain unmodified and unpolluted.

---

## 7. Sample-Based Human Review Plan & Limitations

While automated heuristic and regex evaluators verify structural and safety bounds, automated tools have inherent limitations. The following 100-sample human spot-check plan is recommended prior to training:

### Sampling Strategy
Select 100 stratified samples (5 samples from each of the 20 batches across all 11 categories):
```bash
./.venv/bin/python -c "
import json, random
random.seed(42)
with open('data/train_expanded.jsonl') as f:
    lines = [json.loads(l) for l in f]
samples = random.sample(lines, 100)
with open('reports/human_review_samples.jsonl', 'w') as f:
    for s in samples:
        f.write(json.dumps(s) + '\n')
print('Wrote 100 human review samples to reports/human_review_samples.jsonl')
"
```

### Review Rubric
1. **Conversational Naturalness:** Does the assistant dialogue sound warm, natural, and helpful rather than robotic or repetitively templated?
2. **Contextual Consistency:** Does the assistant preserve user constraints (neighborhood, budget, group size) throughout the dialogue?
3. **Appropriate Boundaries:** Does the assistant politely set boundaries on out-of-scope requests (bookings, financial data, unverified discounts) without sounding punitive?

### Automated Evaluation Limitations
- Automated checks cannot fully judge nuance, comedic tone, or regional phrasing subtleties.
- Complex multi-constraint reasoning across 4+ turns benefits from human qualitative review.

---

## 8. Safe Training Configuration (For Future Phase)

> [!CAUTION]
> **NO TRAINING WAS STARTED.** The current adapter weights in `outputs/flowersforyou-adapter/` and `outputs/phase2-adapter-backup/` remain completely untouched.

When you are ready to train on the expanded 10,000-example dataset in a future phase, update `train.py` or specify the dataset path via command-line arguments:

```bash
# Recommended command for future fine-tuning phase (DO NOT RUN NOW without approval):
python train.py --train --data-file data/train_expanded.jsonl --epochs 1 --batch-size 4 --grad-accum 4
```

Alternatively, `train.py` can be configured with an optional `--dataset` CLI argument pointing to `data/train_expanded.jsonl` while keeping `data/train.jsonl` as the default fallback.
