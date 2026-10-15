# FlowersForYou Fine-Tuning Starter

This is a **prepare-and-validate first** project. It does not start training automatically.

## What this project does
- Validates conversational JSONL datasets.
- Checks Python, hardware, and installed package availability.
- Provides a standalone model-loading and inference verification script (`test_model_load.py`).
- Provides a PEFT LoRA training entry point that requires an explicit `--train` flag (`train.py`).
- Provides comparative evaluation against held-out examples (`evaluate.py`).
- Provides conversational inference for the base model and fine-tuned adapter (`inference.py`).

## Important
- Training is not run by setup or validation commands.
- The included examples are synthetic starter examples, not a production-quality dataset.
- Verified model for Apple Silicon local experimentation: `Qwen/Qwen2.5-0.5B-Instruct` (Apache 2.0 license).
- Do not place private user conversations in training data without appropriate consent and privacy safeguards.
- Store each user's preferences in your application database, not in model weights.
- Real venue search, availability, matching, booking, and payments must be performed and verified by backend services.

## Project layout

```text
flowersforyou-finetuning/
├── data/
│   ├── train.jsonl
│   └── eval.jsonl
├── outputs/
│   └── flowersforyou-adapter/     (created after running train.py --train)
├── test_model_load.py             (Phase 1 verification script)
├── inspect_environment.py         (Environment preflight)
├── validate_data.py               (Dataset schema validator)
├── train.py                       (SFT LoRA trainer with --train safety gate)
├── evaluate.py                    (Held-out comparative evaluation)
├── inference.py                   (Interactive/prompt inference)
├── requirements.txt
└── README.md
```

## 1. Create and activate a virtual environment (Mac/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If your Mac cannot support the required training configuration, use a compatible cloud GPU environment instead.

## 2. Install training dependencies

```bash
python -m pip install -r requirements.txt
```

This installs libraries but does not train the model.

## 3. Inspect the environment

```bash
python inspect_environment.py
```

## 4. Validate datasets

```bash
python validate_data.py
```

Expected: both JSONL files parse and each record contains a valid conversational `messages` list.

## 5. Verify Model Loading & Inference (Phase 1)

```bash
python test_model_load.py
```

Loads `Qwen/Qwen2.5-0.5B-Instruct` onto Apple Silicon MPS in `bfloat16` and tests short chat generation.

## 6. Training (Explicit Opt-In Required)

```bash
# Pre-flight checks only (will NOT train):
python train.py

# Explicit training run:
python train.py --train
```

Without `--train`, the script only performs preflight checks and exits. Do not run `--train` until you are ready to consume compute.

## 7. Comparative Evaluation

```bash
# Evaluate base model only (before training adapter):
python evaluate.py --base-only

# Evaluate base model vs trained adapter (after running train.py --train):
python evaluate.py
```

## 8. Inference

```bash
# Test inference with base model:
python inference.py --base-only --prompt "Can you suggest a quiet coffee shop in Indiranagar?"

# Test inference with fine-tuned adapter:
python inference.py --role planner
```

## Dataset format

One JSON object per line, with `messages` containing `system`, `user`, and `assistant` messages. Each line must be valid JSON. Ensure examples represent both onboarding and planner behavior and are consistent with the selected model's chat template.
