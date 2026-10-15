# FlowersForYou Phase 6D — Preference Extraction Hardening Report

**Project:** `flowersforyou-finetuning` / `~/Desktop/foryou`  
**Date:** October 15, 2026  
**Status:** Completed & Validated (All Tests Passing: 100%)  
**Production Gate:** Phase 6D adapter preserved in isolation; cloud-model default configuration maintained.

---

## Executive Summary

During initial integration validation of the fine-tuned FlowersForYou Phase 6D adapter (`Qwen2.5-0.5B-Instruct` + LoRA adapter `checkpoint-625`), the model exhibited a 40% failure rate (2/5 passed) across the 5 canonical preference extraction benchmarks (`PREF-01` to `PREF-05`):
- `PREF-03` failed due to model hallucination of synthetic placeholders (`city_a`, `city_b`, `budget_inr_max: 100`) when the user had explicitly disclaimed having any location or budget.
- `PREF-04` failed because feeding raw multi-turn dialogue into the prompt caused the small 0.5B model to switch into conversational chat mode, omitting the structured ```json block.
- `PREF-05` failed because the model extracted only the delta interest from the latest utterance without accumulating previously stated preferences into the emitted JSON array.

Following the requirements of Phase 6D Preference Extraction Hardening, minimal, safe application-side schema bridges and context isolation safeguards were implemented in `~/Desktop/foryou`. **All three failed tests have been completely resolved**:
- **Preference Hardening Benchmark (`PREF-01` – `PREF-05`):** **5/5 Passed (100.0%)** (was 2/5).
- **Hardened Regression Suite (`tests/preference-hardening.test.ts`):** **40/40 Assertions Passed (100.0%)**.
- **Existing Application Test Suite (`tests/prototype-test.ts`):** **33/33 Tests Passed (100.0%)** with zero regressions.

---

## 1. Files Changed

All modifications were applied strictly to `~/Desktop/foryou` without modifying training data or retraining the model:

| File Path | Description of Changes |
| :--- | :--- |
| `src/ai/schemas/aiSchemas.ts` | **Task 1 & Task 3 (Schema Bridge & Placeholder Rejection):**<br>• Added `isFabricatedPlaceholder()` and `PLACEHOLDER_PATTERNS` regex matching (`city_a`, `city_b`, `unknown`, `n/a`, `placeholder`, `dummy`, `tbd`, etc.).<br>• Implemented bidirectional bridge in `SchemaValidator.validateMemoryExtraction` translating Phase 6D action payloads (`save_preferences`, `interests`, `preferred_neighborhoods`, `budget_inr_max`) into canonical `StructuredMemoryRecord[]`.<br>• Added validation disallowing budget persistence when user explicitly disclaims budget constraints ("no budget", "haven't decided on a budget").<br>• Added explicit clearing detection (`status: 'superseded'`).<br>• Added candidate value grounding to prevent ungrounded topic hallucinations while preserving legitimate user entities. |
| `src/ai/orchestrator/memoryManager.ts` | **Task 2 (Context Isolation & Reconciliation):**<br>• Added handling in `reconcileAndSave` for incoming records marked `status: 'superseded'`, allowing explicit clearings and corrections to supersede active records in the database.<br>• Preserved existing conflict resolution: newer explicit statements replace older ones, marking previous budget and neighborhood records as `superseded` while keeping unchanged preferences active. |
| `src/ai/memory/memoryExtractor.ts` | **Task 2 (Context Isolation):**<br>• Updated `MemoryExtractor.extract` to isolate extraction prompts, passing the latest user utterance and relevant active preference summaries rather than undifferentiated raw multi-turn dialogue.<br>• Enhanced deterministic fallback parser with regex support for numeric INR budgets, neighborhood corrections (e.g., "scratch Whitefield -> Koramangala"), and explicit preference removals. |
| `package.json` | **Runtime Stability & Test Integration:**<br>• Updated `scripts.test` to use `node --import tsx` to bypass Node.js v26 synchronous IPC spawn timeout bug.<br>• Added convenience script `"test:pref:hardening"`: `DATABASE_MODE=local AI_ENV=test node --import tsx tests/preference-hardening.test.ts`. |
| `tests/preference-hardening.test.ts` | **Task 4 (Regression Suite):**<br>• New comprehensive test suite covering all 9 required verification scenarios (40 total assertions). |

---

## 2. Automated Test Execution & Pass/Fail Counts

### 2.1 Preference Hardening Regression Suite (`tests/preference-hardening.test.ts`)
Executed with: `DATABASE_MODE=local AI_ENV=test node --import tsx tests/preference-hardening.test.ts`

| Test Scenario | Assertions | Status | Notes |
| :--- | :---: | :---: | :--- |
| **1. Normal interest & neighborhood extraction** | 5 | **PASSED** | Extracts `specialty_coffee`, `pottery`, and `Indiranagar` as explicit active records |
| **2. Numeric INR budget extraction** | 4 | **PASSED** | Extracts canonical `budget_ceiling_inr` with value `'1500'` (0.95 confidence) |
| **3. Missing preferences & placeholder rejection** | 8 | **PASSED** | Rejects `city_a`, `city_b`, `unknown`, `N/A`, and disclaimed budget |
| **4. Correcting saved budget (₹800 -> ₹1500)** | 5 | **PASSED** | Active budget updated to `1500`; previous `800` record superseded |
| **5. Adding interest without losing existing** | 5 | **PASSED** | Retains `Outdoor Cycling` & `Sourdough Baking` when adding `Specialty Coffee` |
| **6. Explicitly clearing a preference** | 3 | **PASSED** | Budget cleared; active budget empty; interests remain intact |
| **7. Rejection of invalid JSON & field types** | 2 | **PASSED** | Non-object and malformed types safely rejected without database mutation |
| **8. Preservation of omitted fields** | 3 | **PASSED** | Updating location preserves independent interests and budget records |
| **9. Planner integration with updated preferences**| 5 | **PASSED** | Planning context and generated plan reflect updated location and budget |
| **Total** | **40 / 40** | **100% PASS** | Zero failures |

### 2.2 Baseline Application Test Suite (`tests/prototype-test.ts`)
Executed with: `npm test` (`DATABASE_MODE=local AI_ENV=test node --import tsx tests/prototype-test.ts`)

| Suite | Tests Executed | Passed | Status |
| :--- | :---: | :---: | :---: |
| Baseline Regression Suite (1–8) | 8 | 8 | **100% PASS** |
| Hardened Reasoning & Edge-Case Suite (A–Q) | 17 | 17 | **100% PASS** |
| Phase 3 Production Controls Suite (R–X) | 8 | 8 | **100% PASS** |
| Real-Language Dataset Benchmark (M1–M4, I1–I4) | 8 | 8 | **100% PASS** |
| **Total** | **33 / 33** | **33** | **100% PASS** |

---

## 3. Before / After Results for PREF-01 Through PREF-05

The 5 canonical Phase 6D preference extraction scenarios were re-evaluated against the live model (`Qwen2.5-0.5B-Instruct` + `outputs/flowersforyou-adapter-10k`) through the hardened pipeline:

| Test ID | Scenario Name | Before Hardening | After Hardening | Latency | Key Difference |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **PREF-01** | Interests & Preferred Neighborhoods | **PASS** | **PASS** | 3.91s | Clean extraction of specialty coffee, artisanal pottery, and Indiranagar. |
| **PREF-02** | Numeric INR Budget & Neighborhood | **PASS** | **PASS** | 3.33s | Accurately extracts numeric `budget_inr_max: 1500` in Bandra West for indie board games. |
| **PREF-03** | Missing Preferences (No Hallucinations) | **FAIL** | **PASS** | 3.33s | **Before:** Raw model hallucinated `city_a`, `city_b`, and `budget: 100`.<br>**After:** Placeholder filter dropped `city_a`/`city_b`; disclaimer check dropped budget; only legitimate `Modern Strategy Board Games` persisted. |
| **PREF-04** | Correcting a Stated Preference | **FAIL** | **PASS** | 3.32s | **Before:** Multi-turn conversational history caused 0.5B model to omit ```json block.<br>**After:** Context isolation provided clean utterance; model emitted valid update; Whitefield superseded; Koramangala activated; budget updated to ₹1200. |
| **PREF-05** | Multiple Preferences in Multi-Turn | **FAIL** | **PASS** | 4.12s | **Before:** Multi-turn output emitted only delta interest, omitting previous interests from the array.<br>**After:** MemoryManager merged new interest (`Specialty Coffee Tasting`) with existing profile, preserving `Outdoor Cycling` & `Sourdough Baking`. |
| **Overall** | **Benchmark Accuracy** | **2 / 5 (40%)** | **5 / 5 (100%)** | **3.60s avg** | **+60.0% Absolute Improvement** |

---

## 4. Example Inputs and Validated Outputs

### Example 1: Preference Correction (Budget & Location)
**Context:** User previously had saved `Whitefield` and `budget: 500`.  
**Latest User Statement:**
```text
"Actually, scratch Whitefield—I moved to Koramangala now, and please raise my maximum budget to ₹1200."
```
**Model Output:**
```json
{
  "action": "save_preferences",
  "preferred_neighborhoods": ["Koramangala"],
  "budget_inr_max": 1200
}
```
**Validated `StructuredMemoryRecord[]`:**
```json
[
  {
    "user_id": "test_pref_user",
    "category": "location",
    "key": "neighborhood_koramangala",
    "value": "Koramangala",
    "source": "explicit",
    "confidence": 0.95,
    "status": "active"
  },
  {
    "user_id": "test_pref_user",
    "category": "budget",
    "key": "budget_ceiling_inr",
    "value": "1200",
    "source": "explicit",
    "confidence": 0.95,
    "status": "active"
  }
]
```
**Reconciled User Memory Summary (`getActiveMemorySummary`):**
```json
{
  "location_preferences": ["Koramangala"],
  "budget_preferences": ["1200"]
}
```
*(Previous records for Whitefield and ₹500 budget are transitioned to `status: 'superseded'` in the database and omitted from active planner context.)*

---

### Example 2: Preference Addition (Preserving Existing Interests)
**Context:** User has active interests `["Outdoor Cycling", "Sourdough Baking"]`.  
**Latest User Statement:**
```text
"Also add specialty coffee tasting to my interests, keep it around ₹800 max in Hauz Khas Village."
```
**Model Output:**
```json
{
  "action": "save_preferences",
  "interests": ["specialty_coffee_tasting"],
  "preferred_neighborhoods": ["Hauz Khas Village"],
  "budget_inr_max": 800
}
```
**Validated `StructuredMemoryRecord[]`:**
```json
[
  {
    "user_id": "test_pref_user",
    "category": "interest",
    "key": "specialty_coffee",
    "value": "Specialty Coffee Tasting",
    "source": "explicit",
    "confidence": 0.95,
    "status": "active"
  },
  {
    "user_id": "test_pref_user",
    "category": "location",
    "key": "neighborhood_hauz_khas_village",
    "value": "Hauz Khas Village",
    "source": "explicit",
    "confidence": 0.95,
    "status": "active"
  },
  {
    "user_id": "test_pref_user",
    "category": "budget",
    "key": "budget_ceiling_inr",
    "value": "800",
    "source": "explicit",
    "confidence": 0.95,
    "status": "active"
  }
]
```
**Reconciled User Memory Summary (`getActiveMemorySummary`):**
```json
{
  "interests": ["Outdoor Cycling", "Sourdough Baking", "Specialty Coffee Tasting"],
  "location_preferences": ["Hauz Khas Village"],
  "budget_preferences": ["800"]
}
```
*(All 3 interests are preserved and active.)*

---

### Example 3: Explicit Preference Clearing
**Context:** User has an active budget ceiling of `1500`.  
**Latest User Statement:**
```text
"Please clear my budget limit, I don't want any budget ceiling."
```
**Validated `StructuredMemoryRecord[]`:**
```json
[
  {
    "user_id": "test_pref_user",
    "category": "budget",
    "key": "budget_ceiling_inr",
    "value": "cleared",
    "source": "explicit",
    "confidence": 1.0,
    "status": "superseded"
  }
]
```
**Reconciled User Memory Summary (`getActiveMemorySummary`):**
```json
{
  "budget_preferences": []
}
```
*(Active budget ceiling removed; all other preferences remain active.)*

---

## 5. Remaining Limitations

1. **Sub-Billion Parameter Sensitivity to Prompt Framing:**  
   Because `Qwen2.5-0.5B-Instruct` has only ~500M parameters, it remains susceptible to persona drift if raw multi-turn conversational dialogue is directly injected into the extraction prompt. Context isolation (passing only the latest user statement and structured summary) is required to guarantee reliable JSON extraction.
2. **Compound Multi-Entity Extraction Nuance:**  
   When a user combines multiple disparate operations in a single complex clause (e.g., "Add cycling, remove badminton, but keep badminton on weekends, and raise budget only if we go to Koramangala"), the 0.5B model may struggle with conditional logic. The application's candidate grounding and conversational clarification flows handle these gracefully.
3. **Model Adapter Isolation:**  
   The Phase 6D adapter is maintained in `outputs/flowersforyou-adapter-10k`. The production configuration in `~/Desktop/foryou/src/config/environment.ts` remains pointing to its default cloud model provider (`DATABASE_MODE=local`, `AI_ENV=test`). The Phase 6D adapter is not enabled by default in production.

---

## Conclusion

Phase 6D Preference Extraction Hardening has been successfully implemented and empirically verified. All 3 previously failed extraction tests have passed, 100% of the 40 regression assertions and 33 existing prototype tests passed without any regressions, and existing memory reconciliation behavior was fully preserved.
