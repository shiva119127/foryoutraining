# FlowersForYou 10,000-Example Fine-Tuning Dataset Plan

**Target Dataset:** `data/train_expanded.jsonl` (10,000 accepted training records)  
**Sidecar Provenance Manifest:** `data/dataset_manifest.jsonl` (10,000 provenance records)  
**Batch Storage:** `data/generated_batches/batch_01.jsonl` through `data/generated_batches/batch_20.jsonl` (500 records each)  
**Batch Audit Reports:** `reports/dataset_batches/batch_01_report.md` through `batch_20_report.md`  

---

## 1. Category Allocation & Rationale

| # | Behavioral Category | Total Target | Per-Batch Target (20 Batches) | Key Behavioral Objective |
|---|---------------------|---:|---:|--------------------------|
| 1 | **First-time onboarding & progressive discovery** | 800 | 40 | Low-friction discovery; 1 question at a time; no rigid questionnaires; friendly welcoming tone. |
| 2 | **Activity discovery & personalized planning** | 1,300 | 65 | Suggesting activities tailored to user vibe (creative, social, outdoor, relaxed); gathering missing requirements. |
| 3 | **Structured preference extraction** | 700 | 35 | Strict canonical JSON schema (`action: save_preferences`, `interests`, `preferred_neighborhoods`, numeric `budget_inr_max`). Value preservation. |
| 4 | **Returning users & saved preference context** | 600 | 30 | Warm greeting without restarting onboarding; acknowledging supplied profile context; immediate planning invitation. |
| 5 | **Budget, location, group size, timing & accessibility** | 900 | 45 | Factoring in Indian urban context (Bengaluru, Mumbai, Delhi-NCR, Pune, Hyderabad), INR budgets, solo/group dynamics, timings, and mobility needs. |
| 6 | **Multi-turn conversations, corrections & changing preferences** | 1,200 | 60 | Handling user pivots ("Actually, make it pottery instead", "Increase budget to ₹1500", "Change to Saturday"); preserving context across turns. |
| 7 | **Undecided users, skipped onboarding & low-friction starts** | 700 | 35 | Offering 2–3 concrete contrasting activities to undecided users; respecting skip commands with instant activity planning questions (no surveys). |
| 8 | **Tool-grounded venue search, results & verified facts** | 1,000 | 50 | Using `role: "tool"` inputs to ground factual recommendations (prices, hours, amenities, partner status); clearly attributing data to tool findings. |
| 9 | **Booking safety, unavailable venues, prices & partner verification** | 1,400 | 70 | Refusing to fabricate confirmations or voucher IDs; explaining live verification requirements; refusing to invent partner discounts without tool confirmation. |
| 10 | **Ambiguous requests, refusals, edge cases & privacy** | 700 | 35 | Handling missing locations/budgets gracefully; refusing sensitive PII collection; refusing unrealistic or out-of-scope requests safely. |
| 11 | **User feedback, failed plans & recovery** | 700 | 35 | Handling user feedback ("The cafe was too noisy", "The event got cancelled", "We didn't like the food"); empathetic recovery and alternative suggestions. |
| | **Total** | **10,000** | **500** | **Full conversational and safety coverage** |

---

## 2. Behavioral & Safety Contract

Every generated example must adhere to the core FlowersForYou safety contract:
1. **Never promise or confirm reservations without tool results**: No `"Your table is confirmed"`, `"Confirmation generated"`, `"We'll ensure it's booked"`, or `"I can provide that"`.
2. **Never invent commercial discounts or partner status**: Third Wave Coffee, Blue Tokai, etc., cannot be claimed as partners or offering member discounts unless confirmed in a `tool` message.
3. **Never invent specific opening hours, transit routes, or travel times**: No `"open daily from 9 AM to 11 PM"` or `"take bus 201"` asserted as verified facts without tool grounding.
4. **Canonical preference schema fidelity**:
   ```json
   {
     "action": "save_preferences",
     "interests": ["..."],
     "preferred_neighborhoods": ["..."],
     "budget_inr_max": 750
   }
   ```
   No array roots, no nested wrappers, no non-INR strings.
5. **No profiling restarts**: Returning users get warm greetings; skip-onboarding users get immediate planning questions.
6. **Undecided user contrast**: Must offer 2–3 concrete contrasting options (e.g. coffee tasting vs pottery class vs outdoor bouldering).
7. **Zero held-out leakage**: Prompts and responses must have strictly $\text{Jaccard} < 0.40$ against `data/eval.jsonl` (9 held-out examples).

---

## 3. Linguistic & Contextual Diversity Matrix

To ensure natural, non-templated dialogue across 10,000 records:
- **Urban Indian Contexts**: Bengaluru (Indiranagar, Koramangala, Jayanagar, Whitefield, HSR Layout, Malleshwaram, Church Street, JP Nagar, Sadashivanagar), Mumbai (Bandra, Colaba, Juhu, Lower Parel), Delhi-NCR (Hauz Khas, Connaught Place, CyberHub, Saket), Pune (Koregaon Park, Baner), Hyderabad (Jubilee Hills, Gachibowli).
- **Currencies & Budgets**: Realistic INR pricing from ₹200 (budget cafe/tea) to ₹5,000+ (curated workshops/fine dining outings).
- **Tone & Formality**: Ranging from casual modern slang ("hey, super bored today", "need chill plans") to structured family/group planning ("planning an anniversary dinner for 6").
- **Turn Depth**: From 2-turn single-exchange prompts to 4-turn and 6-turn detailed conversational progressions with tool results and user preference pivots.
- **Vibe Diversity**: Solo introspective outings, romantic dates, active sports, creative arts/crafts, board games, culinary/coffee, outdoor parks, team outings.

---

## 4. Batch Generation & Validation Pipeline

1. **Batch Generation**:
   `scripts/generate_dataset.py` generates 20 batches of candidates using a deterministic random seed per batch (`seed = 1000 + batch_num`).
2. **Quality Validation**:
   `scripts/validate_dataset.py` inspects each batch through a 10-point check suite:
   - JSON parsing & structural schema integrity (`messages` format, allowed roles, assistant finale).
   - Canonical JSON schema enforcement on extraction examples.
   - Exact value & type fidelity (INR numeric budget, interest preservation).
   - Semantic safety rule execution (refusal of unverified booking promises, discounts, fake hours).
   - Tool-grounded consistency check (attributing facts only when `tool` role is present).
   - Exact & normalized duplicate detection.
   - Intra-batch and cross-batch near-duplicate check (Jaccard $< 0.85$).
   - Held-out evaluation leakage check against `data/eval.jsonl` (zero exact matches, Jaccard $< 0.40$).
   - Provenance manifest synchronization.
3. **Compilation**:
   Once all 20 batches pass, they are compiled into `data/train_expanded.jsonl` (10,000 lines) and `data/dataset_manifest.jsonl` (10,000 provenance records).
