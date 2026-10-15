# FlowersForYou Phase 6C — Dataset Diversity Repair & Training Readiness Report

**Generated:** 2026-10-15  
**Project:** `flowersforyou-finetuning`  
**Phase:** 6C (Repaired Dataset Candidate & Independent Quality Audit)  
**Status:** VALIDATION COMPLETE — READY FOR TRAINING REVIEW (NO TRAINING EXECUTED)  

---

## 1. Executive Summary

Phase 6C successfully resolved the critical dataset quality issues identified in the Phase 6B audit. The synthetic fine-tuning corpus has been systematically regenerated and compiled into `data/train_expanded_v2.jsonl` (10,000 records) and synchronized 1-to-1 with `data/dataset_manifest_v2.jsonl`.

### Key Outcomes:
1. **Near-Duplicate Elimination:** Reduced global cross-batch near-duplicate pairs ($J \ge 0.85$) from **8,973 to 0**. Near-duplicates at $J \ge 0.80$ are also **0**. The maximum internal user prompt Jaccard across all 10,000 records is **0.7955**.
2. **Zero Evaluation Leakage:** Retained strict isolation from both evaluation benchmarks (`data/eval.jsonl` [9 records] and `data/eval_expanded.jsonl` [56 records]). Maximum train-to-eval Jaccard is **0.3784** (well within the $< 0.40$ threshold); zero exact prompt or response overlaps.
3. **Grammatical Integrity:** 100% elimination of plural/gerund phrasing artifacts (e.g., *"a leather craft basics"*, *"a vintage arcade games"*) through countable noun heads (`workshop`, `session`, `class`, `walk`).
4. **Context Continuity on Skip-Onboarding:** Assistant responses retain user-stated location context upon skipping onboarding without re-asking for information already provided.
5. **Canonical Schema Compliance:** 100% compliance across all 700 structured preference extraction records using the canonical `action: save_preferences` schema and numeric INR budgets.
6. **Strict Quality-Controlled Volume:** Exactly **10,000 high-quality records** accepted without padding or boilerplate repetition, fully preserving all 11 behavioral categories.
7. **Absolute Immutability Verified:** Zero training was executed. All baseline evaluation files (`data/eval.jsonl`, `data/eval_expanded.jsonl`), immutable source datasets (`data/train.jsonl`, `data/train_expanded.jsonl`), and model adapter weights remain byte-for-byte identical.

---

## 2. Investigation of Repetition & Duplication

### 2.1 The 8,973 Figure Explained
In Phase 6B, the audit reported 8,973 cross-batch near-duplicate pairs. Our code inspection of `scripts/audit_expansion.py` and verification using `scripts/investigate_duplicates.py` confirmed:
- **Definition:** The 8,973 figure represents the number of **unique unordered pairs of records $(i, j)$** with $1 \le i < j \le 10,000$ where the user prompt token sets share a **Jaccard similarity $\ge 0.85$**.
- **Impacted Record Count:** Exactly **3,919 distinct records** (39.19% of the 10,000 dataset) were entangled in near-duplicate clusters.
- **Cluster Structure:** Repetition was not confined within individual batches; identical sentence frames reappeared across all 20 batches because the original generator lacked a cross-batch global deduplication index.

### 2.2 Root Cause Analysis
1. **Template Brittleness:** In `scripts/generate_dataset.py` (Phase 6A), user prompts were built from fixed 14–20 word sentence frames where only a single token (`{neighborhood}` or `{activity}`) was altered (e.g., *"Looking for a pottery workshop in {area} under {budget} for two people"*). Because the common token prefix and suffix accounted for 85–95% of total words, Jaccard similarity routinely exceeded 0.85.
2. **Assistant Response Monoculture:** In categories such as `first_time_onboarding` and `undecided_and_skip_onboarding`, assistant responses followed fixed phrasing patterns with minimal persona or lexical diversity.
3. **Neighborhood-Only Perturbation:** Varying only the geographical entity (e.g., "Koramangala" vs "Indiranagar") failed to introduce semantic or syntactic independence.

### 2.3 Semantic Embedding Analysis
Using pooled hidden states from `Qwen/Qwen2.5-0.5B-Instruct` (the base model), we computed cosine similarity across prompt variations:
- **Lexical near-duplicates with swapped locations:** Cosine similarity = **0.9952 to 0.9994** (functionally identical representations in model embedding space).
- **Semantically distinct conversational intents:** Cosine similarity = **0.8400 to 0.8713**.
This proved conclusively that neighborhood-only variations do not provide diverse training signals to the model.

---

## 3. Repair Methodology & Architecture

The repaired dataset generator was developed from the ground up in `scripts/generate_dataset_v2.py`.

```
+-----------------------------------------------------------------------------------+
|                        Repaired Generation Architecture                           |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [Combinatorial Clause Engine]                                                    |
|    - 12+ Openers x 8 Personas x 10 Motivations x 15 Constraints x 10 Closers       |
|    - Grammatically Sound Head Nouns (workshop, session, masterclass, walk)        |
|                                                                                   |
|                                     |                                             |
|                                     v                                             |
|                                                                                   |
|  [Zero-Leakage Rejection Filter]                                                  |
|    - Checks against 65 held-out eval prompts (eval.jsonl & eval_expanded.jsonl)   |
|    - Strict Jaccard threshold: Reject if J >= 0.38 (Target: < 0.40)               |
|                                                                                   |
|                                     | (Pass)                                      |
|                                     v                                             |
|                                                                                   |
|  [Global Inverted Diversity Index]                                                |
|    - Token inverted index over all previously accepted records across all batches |
|    - Strict Diversity threshold: Reject if J >= 0.80 against ANY accepted record  |
|                                                                                   |
|                                     | (Pass)                                      |
|                                     v                                             |
|                                                                                   |
|  [Behavioral & Schema Validators]                                                 |
|    - Preference Extraction: Canonical schema (action: save_preferences)           |
|    - Skip-Onboarding: Zero context amnesia (incorporates prior area)              |
|    - Honest Grounding: Partner & availability verification clauses                |
|                                                                                   |
|                                     | (Pass)                                      |
|                                     v                                             |
|                                                                                   |
|  [Accepted Candidates: data/train_expanded_v2.jsonl + Manifest]                   |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

### 3.1 Combinatorial Clause Generation
Instead of monolithic sentence templates, user prompts are constructed dynamically by sampling across modular syntactic components:
- Contextual opening phrasing (greeting, casual, direct, inquiring)
- User persona / demographic context (college student, busy professional, couple, expat, retiree)
- Motivation / occasion (anniversary, team hangout, solo decompression, creative hobby)
- Countable activity nouns (ensuring proper gerund and plural syntax)
- Explicit constraints (budget in INR, dietary, pet-friendly, wheelchair accessibility)
- Conversational closers

### 3.2 Global Inverted Diversity Index (`DiversityIndex`)
To guarantee global diversity across all 20 batches without prohibitive $O(N^2)$ pairwise latency:
- A token-to-record inverted index maintains all accepted user prompts.
- For every generated candidate, candidates sharing $\ge 3$ rare tokens are retrieved and evaluated for exact Jaccard similarity.
- Any candidate with **Jaccard similarity $\ge 0.80$ against ANY previously accepted record** is discarded and regenerated.

### 3.3 Zero-Leakage Held-Out Separation
All 65 held-out evaluation scenarios (9 in `data/eval.jsonl`, 56 in `data/eval_expanded.jsonl`) were loaded into memory prior to generation. Any candidate prompt with Jaccard $\ge 0.38$ against any evaluation scenario was immediately rejected.

### 3.4 Specific Defect Corrections
1. **Plural & Gerund Activity Phrasing:** Added proper countable noun heads (`workshop`, `session`, `class`, `walk`, `tour`, `masterclass`). The ungrammatical *"a leather craft basics"* is replaced with *"a leather craft basics workshop"*, and *"a vintage arcade games"* is replaced with *"a vintage arcade gaming session"*.
2. **Skip-Onboarding Context Continuity:** When a user previously mentioned an area (e.g., *"Skip the questions, just show me spots around Indiranagar"*), the assistant explicitly acknowledges the area: *"Since you mentioned Indiranagar earlier, let's bypass the survey and dive straight into recommendations there..."*
3. **Preference Extraction Canonical Schema:** All 700 structured preference examples strictly emit ````json ... ```` containing the required canonical schema:
   ```json
   {
     "action": "save_preferences",
     "interests": ["pottery", "live jazz"],
     "preferred_neighborhoods": ["Indiranagar"],
     "budget_inr_max": 1200,
     "budget_inr_min": 500,
     "timing": "weekend_evenings",
     "social_vibe": "lively",
     "max_group_size": 2
   }
   ```
4. **Honest Tool Grounding & Verification:** Assistant responses for venue availability, booking execution, pricing, and partner discounts explicitly state that live verification against partner database records is required.
5. **Provenance Transparency:** Every record in `data/train_expanded_v2.jsonl` maps directly to an entry in `data/dataset_manifest_v2.jsonl` detailing `batch_id`, `category`, `generation_method: synthetic_combinatorial_v2`, `temperature`, `seed`, `record_index`, and timestamp.

---

## 4. Quality Audit & Comparative Metrics

The repaired dataset was independently audited using `scripts/validate_dataset.py` and `scripts/audit_v2_dataset.py`.

### 4.1 Global Metric Comparison

| Metric | Phase 6A/6B (v1: `train_expanded.jsonl`) | Phase 6C (v2: `train_expanded_v2.jsonl`) | Status |
| :--- | :--- | :--- | :--- |
| **Total Records** | 10,000 | **10,000** | Preserved |
| **Manifest Synchronization** | 10,000 | **10,000** | In Sync (1:1) |
| **Exact Full Conversation Duplicates** | 0 | **0** | PASSED |
| **Exact User Prompt Duplicates** | 0 | **0** | PASSED |
| **Normalized User Prompt Duplicates** | 0 | **0** | PASSED |
| **Cross-Batch Near-Duplicate Pairs ($J \ge 0.85$)** | **8,973** | **0** | **RESOLVED (-100%)** |
| **Near-Duplicate Pairs ($J \ge 0.80$)** | > 15,000 | **0** | **RESOLVED (-100%)** |
| **Max Internal User Prompt Jaccard** | 0.9412 | **0.7955** | Strictly $< 0.80$ |
| **Max Train-to-Eval Jaccard** | 0.3889 | **0.3784** | Safe ($< 0.40$) |
| **Held-Out Evaluation Leakages ($J \ge 0.38$)** | 0 | **0** | Zero Leakage |
| **Grammar Phrasing Artifacts** | High (prevalent in batches) | **0** (0.0%) | **RESOLVED** |
| **Skip-Onboarding Context Amnesia** | High (frequent re-asking) | **0** (0.0% in 115 audits) | **RESOLVED** |
| **Preference Schema Errors** | Present in baseline | **0** (0.0% in 700 audits) | **RESOLVED** |
| **Evaluator Safety Violations** | 0 | **0** (0.0%) | **PASSED (10,000/10,000)** |

### 4.2 Category Allocation Comparison

The planned distribution across all 11 behavioral categories is preserved down to the single record:

| Category | v1 Count | v2 Count | Delta | Target Coverage Description |
| :--- | :---: | :---: | :---: | :--- |
| `activity_discovery` | 1,300 | 1,300 | +0 | Broad exploration, contrasting options |
| `ambiguous_and_privacy` | 700 | 700 | +0 | Privacy defense (cards, PAN, keys, addresses) |
| `booking_safety_and_verification` | 1,400 | 1,400 | +0 | Refusing ungrounded bookings, partner status |
| `constraints_handling` | 900 | 900 | +0 | Dietary, pet-friendly, mobility, budgets |
| `feedback_and_recovery` | 700 | 700 | +0 | Graceful recovery from negative feedback |
| `first_time_onboarding` | 800 | 800 | +0 | 2–3 question conversational onboarding |
| `multi_turn_corrections` | 1,200 | 1,200 | +0 | Mid-flow pivots, date/neighborhood changes |
| `returning_users` | 600 | 600 | +0 | Context recall without restarting survey |
| `structured_preference_extraction` | 700 | 700 | +0 | Canonical `save_preferences` JSON |
| `tool_grounded_search` | 1,000 | 1,000 | +0 | Search results synthesis and citations |
| `undecided_and_skip_onboarding` | 700 | 700 | +0 | Immediate planning, no survey stall |
| **Total** | **10,000** | **10,000** | **+0** | **100.0% Target Achieved** |

---

## 5. 56-Scenario Benchmark Baseline & Documented Failures

The unchanged 56-scenario held-out benchmark (`data/eval_expanded.jsonl`) was established in Phase 6B as the performance hurdle for fine-tuning.

### 5.1 Baseline Results Summary
- **Base Model (`Qwen/Qwen2.5-0.5B-Instruct`):** 48 / 56 passed (85.7% pass rate)
- **Phase 4 LoRA Adapter:** 47 / 56 passed (83.9% pass rate)

### 5.2 Failure Analysis to Address in Next Training Run

Both models exhibited specific systematic failures that the repaired dataset `train_expanded_v2.jsonl` is designed to correct:

1. **Five Structured Preference Extraction Failures (Records #10, #11, #12, #13, #14):**
   - *Base Model & Adapter Failure:* When presented with multi-turn preference summaries (e.g., *"I love indie pottery workshops and jazz cafes around Indiranagar with a budget under ₹1000"*), both models failed 5 / 5 tests.
   - *Failure Modes:*
     - Emitting arbitrary keys (e.g., `"neighborhood"` instead of canonical `"preferred_neighborhoods"`).
     - Missing the mandatory root `"action": "save_preferences"` dispatch key.
     - Extracting budget as formatted string text (`"$1000"`, `"₹1,000"`) instead of integer INR values (`1000`).
     - Omitting explicitly stated interests or wrapping the output in explanatory conversational text outside the markdown JSON block.
   - *Correction in v2:* Exactly 700 clean, canonical schema demonstrations in `train_expanded_v2.jsonl`.

2. **Partner Discount & Membership Hallucinations:**
   - *Failure Mode:* Claiming that venues (e.g., Third Wave Coffee) offer special member discounts without checking live partner records.
   - *Correction in v2:* 1,400 records in `booking_safety_and_verification` instilling honest refusal and live verification queries.

3. **Booking Commitments without Tool Results:**
   - *Failure Mode:* Promising that a table or workshop is "booked" or "confirmed for tonight" when no booking API has been invoked.
   - *Correction in v2:* Systematic grounding rules demonstrating that availability and confirmation must be retrieved from live tool outputs.

4. **Skip-Onboarding Survey Loops:**
   - *Failure Mode:* When a user asks to skip onboarding, the model asks another profiling question or offers vague stalls.
   - *Correction in v2:* 700 records demonstrating immediate transition to 2 concrete contrasting activity choices while preserving user-stated neighborhood context.

---

## 6. Recommended One-Epoch LoRA Experiment

With 10,000 high-diversity, quality-controlled examples, training for multiple epochs risks catastrophic forgetting, excessive memorization of synthetic syntax, and unnecessary compute overhead. A single epoch provides sufficient gradient updates to align conversational behavior.

### 6.1 Hyperparameter Specifications

| Parameter | Recommended Value | Rationale |
| :--- | :--- | :--- |
| **Epochs** | **1 epoch** | 10,000 diverse steps is ample for behavioral alignment; avoids memorization |
| **Effective Batch Size** | **16** (Batch size 2, Grad Accum 8) | Stable gradient updates while remaining within Apple Silicon MPS RAM |
| **Total Optimization Steps** | **625 updates** | Sufficient convergence across all 11 behavioral categories |
| **Learning Rate** | **$1.0 \times 10^{-4}$** | Optimal for Qwen2.5-0.5B LoRA with AdamW |
| **LR Scheduler** | **Cosine with warm-up** | 50 warm-up steps ($\sim 8\%$ of training) followed by cosine decay |
| **LoRA Target Modules** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Full linear coverage for deep behavioral and structural alignment |
| **LoRA Rank & Alpha** | $r = 16, \alpha = 32$ (Dropout = 0.05) | Balances expressive capacity with regularization |
| **Max Sequence Length** | **1,024 tokens** | Captures complete multi-turn conversations and tool traces |
| **Precision** | `bfloat16` on Apple MPS | Native Apple Silicon hardware acceleration |

### 6.2 Success Criteria for Training Run
1. **Benchmark Pass Rate:** $\ge 54 / 56$ on `data/eval_expanded.jsonl` (Target: 100% on safety rules).
2. **Preference Extraction:** $5 / 5$ pass rate on structured JSON extraction scenarios.
3. **Safety Zero-Tolerance:** 0 booking fabrications, 0 ungrounded partner discount claims, 0 privacy leaks.
4. **Generalization:** Retain general conversational tone without template collapse.

---

## 7. Artifact Integrity & Restriction Certification

In strict adherence to the project guidelines and instructions:
- **NO TRAINING WAS RUN:** `train.py --train` was not executed.
- **EXISTING ADAPTERS UNTOUCHED:**
  - `outputs/flowersforyou-adapter/adapter_model.safetensors`: Exactly **17,640,136 bytes** (Unchanged).
  - `outputs/phase2-adapter-backup/adapter_model.safetensors`: Exactly **17,640,136 bytes** (Unchanged).
- **BASELINE EVALUATION DATASETS PRESERVED BYTE-FOR-BYTE:**
  - `data/eval.jsonl`: MD5 `83221b5d36c71b4b50d7439541678d16` (100% MATCH).
  - `data/eval_expanded.jsonl`: MD5 `46c5be6cab32ec808df7ef3c6f13ddaa` (100% MATCH).
- **IMMUTABLE SOURCE DATASETS PRESERVED:**
  - `data/train.jsonl`: MD5 `4edb9c60fe6dd3dfa6d65e40d34b6e69` (100% MATCH).
  - `data/train_expanded.jsonl`: MD5 `7f584fe28d2986d560cdcaf15e4c5ed2` (100% MATCH).
- **NEW OUTPUT ARTIFACTS CREATED SEPARATELY:**
  - Repaired training candidate: `data/train_expanded_v2.jsonl` (10,000 records).
  - Synchronized provenance manifest: `data/dataset_manifest_v2.jsonl` (10,000 records).
  - Generator engine: `scripts/generate_dataset_v2.py`.
  - Comprehensive audit suite: `scripts/audit_v2_dataset.py`.
  - Duplicate investigation script: `scripts/investigate_duplicates.py`.
