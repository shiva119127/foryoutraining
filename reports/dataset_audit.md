# FlowersForYou Fine-Tuning Dataset & Codebase Audit

**Date:** 2026-10-15  
**Target:** 10,000-example fine-tuning dataset expansion  
**Repository:** `flowersforyou-finetuning`  
**Model Architecture:** `Qwen/Qwen2.5-0.5B-Instruct` (LoRA adapter via PEFT/TRL)  
**Execution Environment:** macOS Apple Silicon MPS / Python 3.14.6  

---

## 1. Exact JSONL Structure Expected by `train.py` & TRL

- **File Format:** Standard JSON Lines (JSONL), UTF-8 encoded, exactly one JSON object per line.
- **Root Object Schema:** A JSON dictionary containing a single required top-level key: `"messages"`.
- **Loader Pipeline in `train.py`:**
  ```python
  train_data = load_dataset("json", data_files=str(train_path), split="train")
  ```
  `datasets.load_dataset` parses each JSONL line into a row.
- **TRL Integration:**
  `train.py` initializes `SFTTrainer` (version 0.26.2) with:
  ```python
  trainer = SFTTrainer(
      model=model,
      args=config,
      train_dataset=train_data,
      eval_dataset=eval_data,
      peft_config=lora_config,
      processing_class=tokenizer,
  )
  ```
  In TRL 0.26+, passing `processing_class=tokenizer` with a dataset containing a `"messages"` column triggers the conversational SFT formatting pathway. TRL internally invokes `tokenizer.apply_chat_template(record["messages"], tokenize=False)` and tokenizes up to `max_length=512`.
- **Constraint:** No arbitrary or extra metadata keys (such as `id`, `category`, `provenance`, or `batch`) may be placed directly at the root of training JSONL records, as this can cause schema divergence or feature errors in Hugging Face `datasets` during batch collating. All provenance metadata must reside in a separate sidecar manifest (`data/dataset_manifest.jsonl`).

---

## 2. Message Format, Supported Roles, & Multi-Turn Representation

- **Messages Array:** `record["messages"]` is a list of message objects:
  ```json
  [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
  ```
- **Supported Roles:**
  Verified against `outputs/flowersforyou-adapter/chat_template.jinja`, `test_model_load.py`, and `validate_data.py`:
  1. `"system"`: Sets persona, behavioral rules, constraints, and instructions.
  2. `"user"`: User utterances, requests, and follow-ups.
  3. `"assistant"`: Target model response to be trained or evaluated.
  4. `"tool"`: Grounded backend search results, partner verification responses, or database status.
- **Multi-Turn Representation:**
  Represented as sequential alternating turns within the single `"messages"` array.
  Example from `data/train.jsonl` (Record #36):
  ```json
  {
    "messages": [
      {"role": "system", "content": "You are the FlowersForYou planner..."},
      {"role": "user", "content": "Help me arrange an afternoon picnic at Sankey Tank with 4 friends."},
      {"role": "assistant", "content": "Sankey Tank is a lovely spot... What total budget do you have in mind...?"},
      {"role": "user", "content": "Our budget is ₹1200 total, and we'd like vegetarian snacks."},
      {"role": "assistant", "content": "A ₹1200 budget for 5 people at Sankey Tank is very practical..."}
    ]
  }
  ```
- **Ordering Constraints:**
  - Must begin with a `"system"` or `"user"` message.
  - Conversation must contain at least one `"user"` message.
  - Final message in the array **MUST** have `role: "assistant"` (the training target).
  - All `content` fields must be non-empty strings.

---

## 3. Tool Calls and Tool Results Support

- **Chat Template Implementation:**
  In `Qwen/Qwen2.5-0.5B-Instruct`'s chat template (`chat_template.jinja`):
  ```jinja
  {%- elif message.role == "tool" %}
      {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
          {{- '<|im_start|>user' }}
      {%- endif %}
      {{- '\n<tool_response>\n' }}
      {{- message.content }}
      {{- '\n</tool_response>' }}
      {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
          {{- '<|im_end|>\n' }}
      {%- endif %}
  {%- endif %}
  ```
- **Training Pipeline Support:**
  When a message with `role: "tool"` is passed, `tokenizer.apply_chat_template` wraps it inside `<tool_response> ... </tool_response>` within a user block.
- **Current Dataset Usage:**
  Existing `data/train.jsonl` Record #27 demonstrates grounded tool usage:
  ```json
  {"role": "user", "content": "Are there any verified weekend pottery workshops in HSR Layout under ₹800?"},
  {"role": "tool", "content": "{\"search_results\": [{\"venue\": \"Clay Station Studio\", \"area\": \"HSR Layout\", \"verified_price_inr\": 750, \"verified_slots\": [\"Saturday 2 PM\", \"Sunday 11 AM\"], \"partner_status\": \"verified\"}]}"},
  {"role": "assistant", "content": "Yes! Based on verified search results, Clay Station Studio in HSR Layout offers..."}
  ```
  The assistant response grounds its factual claims (prices, slots, partner status) strictly on the contents of the `tool` message. When no `tool` message is present, factual assertions (rates, hours, booking confirmations) must be refrained from.

---

## 4. Current Canonical Preference-Extraction Schema

Defined in `validate_data.py` (`CANONICAL_PREFERENCE_KEYS`) and verified in `evaluate.py`:

- **Envelope:** Markdown fenced code block: ````json\n{ ... }\n````
- **Root Object:** Must be a JSON Object (dictionary). Array roots or nested wrappers (`{"preferences": {...}}`) are non-canonical and rejected.
- **Required Keys:**
  1. `"action"`: String, strictly equal to `"save_preferences"`.
  2. `"interests"`: List of strings (e.g. `["strategy_board_games", "specialty_coffee"]`). Values must preserve user-stated terms and must not be mutated (e.g. `indie_board_games` must not become `Indianness`).
  3. `"preferred_neighborhoods"`: List of strings (e.g. `["Indiranagar"]`, `["Koramangala"]`, `["Jayanagar"]`).
  4. `"budget_inr_max"`: Numeric integer or float (e.g. `600`, `850`). **Must NOT be a string** (e.g. `"$700"` or `"700"` is rejected; foreign currencies like `$` are rejected).
- **Allowed Canonical Optional Keys:**
  - `"budget_inr_min"`: Numeric integer or float.
  - `"timing"`: List of strings (e.g. `["weekends"]`, `["evenings"]`).
  - `"social_vibe"`: String (e.g. `"solo_or_small_group"`, `"relaxed"`, `"lively"`).
  - `"max_group_size"`: Numeric integer (e.g. `4`).
- **Disallowed / Non-Canonical Keys:**
  `preferences`, `profile`, `genre`, `genre_preference`, `location`, `location_preference`, `budget_limit`, `maximum_budget`.

---

## 5. Validator and Evaluator Coverage & Gaps

### What `validate_data.py` Currently Checks:
1. Valid JSONL syntax per line.
2. Root object contains a valid `messages` list.
3. Every message is a dictionary with `role` in `{"system", "user", "assistant", "tool"}` and non-empty string `content`.
4. Conversation contains at least one `user` turn and ends with an `assistant` turn.
5. Canonical schema validation for `"preference extractor"` dialogues (verifies `action: save_preferences` and absence of non-canonical keys).
6. Cross-dataset leakage: exact prompt match, exact assistant response match, and token Jaccard similarity between `train.jsonl` and `eval.jsonl`.
7. Intra-dataset near-duplicates: flags any pair with token Jaccard similarity $\ge 0.85$.

### What `validate_data.py` Misses:
- Does not check semantic safety rules (unsupported booking promises, fake discounts, ungrounded opening hours) inside the training dataset.
- Does not verify value fidelity in extraction examples (whether budget numbers in the JSON match the user prompt).
- Does not validate tool message payloads for valid JSON or verify that the assistant only makes factual claims when grounded by a tool turn.
- Does not enforce category balance or link records to provenance manifests.

### What `evaluate.py` Currently Checks:
- Direct or implied booking confirmations and provision claims (`"I can provide that"`).
- Unverified promises to book, secure, arrange, or guarantee reservations.
- Invented complimentary perks, drinks, and commercial discounts (including specific percentage discounts and discount plan membership).
- Specific unverified opening hours and operating schedules.
- Non-canonical preference extraction schemas, missing required keys, corrupted interests, and non-numeric budgets.
- Skip onboarding survey restarts vs active planning initiation.
- Undecided user guidance (requires 2–3 concrete contrasting activities; rejects open-ended stalls).
- Returning user greetings (rejects survey restarts).
- Unverified transit routes, shuttle lines, and travel times.

---

## 6. Current Record Counts, Duplicates, and Evaluation Coverage

- **`data/train.jsonl`:** Exactly 36 records.
- **`data/eval.jsonl`:** Exactly 9 records.
- **Duplicate Rates:**
  - Exact duplicates: 0.
  - Intra-train near-duplicates ($\ge 0.85$ Jaccard): 0.
  - Train-to-eval leakage: 0 exact matches. Maximum Jaccard similarity between train and eval is **0.312** (Train #8 vs Eval #2: `"Can you just book a table for 4 in JP Nagar for ₹500 tonight?"` vs `"Book a table at a venue for ₹300 tonight."`).
- **Evaluation Coverage:**
  9 held-out records covering:
  1. Returning user greeting without onboarding restart.
  2. Table booking request with availability/pricing explanation.
  3. Undecided user offering concrete contrasting activities.
  4. Skip onboarding transition directly into planning questions.
  5. Refusal to generate fake booking confirmation vouchers.
  6. Honest refusal of unrealistic constraints (10 PM pottery for ₹300) with practical alternatives.
  7. Multi-turn picnic planning without ungrounded venue names or rigid schedules.
  8. Canonical preference extraction with numeric INR budget.
  9. Refusal to confirm Third Wave Coffee member discount without live partner verification.
- **Schema Inconsistencies:** None in active files. All 36 train and 9 eval records pass all structural and canonical checks.

---

## 7. Status of Consented Real Conversations

- **Repository Search:**
  Audited the entire workspace. Only `data/train.jsonl` and `data/eval.jsonl` exist. There are no customer chat logs, database dumps, exported transcripts, or real conversation files present.
- **Privacy & Compliance Policy:**
  Per project instructions, we do not inspect or search personal directories, Desktop folders, or external services.
- **Design Decision:**
  All 10,000 examples will be generated via a high-diversity synthetic pipeline with controlled scenario families. In the provenance manifest, all records will be clearly marked as:
  - `source_type`: `"synthetic"`
  - `consent_status`: `"not_applicable_synthetic"`
  No synthetic example will be falsely labeled as real.

---

## 8. Protection Against Accidental Overwrites & Model Corruption

1. **Training Gate in `train.py`:**
   Running `python train.py` executes only preflight checks and dataset validation, then immediately exits with:
   `PRE-FLIGHT ONLY: checks passed. Training was NOT started. Review model license, chat template, hardware and this script before using --train.`
   Training requires explicit CLI argument `--train`.
2. **Adapter Protection:**
   The current adapter is stored at `outputs/flowersforyou-adapter/`, with an immutable backup at `outputs/phase2-adapter-backup/`. No script will write to or modify either directory during this dataset expansion task.
3. **Dataset Isolation:**
   - Active training dataset `data/train.jsonl` (36 records) is left completely intact.
   - Held-out evaluation dataset `data/eval.jsonl` (9 records) is strictly preserved and held out.
   - The new 10,000-example dataset will be generated into `data/generated_batches/` and compiled into `data/train_expanded.jsonl` alongside `data/dataset_manifest.jsonl`.
   - The user retains full control over whether and when to switch the active training file.
