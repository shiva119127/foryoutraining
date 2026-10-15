# Phase 6B: Training Readiness & Architecture Verification Report

## Executive Summary

This report evaluates the technical readiness, memory footprint, runtime feasibility, and safety boundaries for fine-tuning the FlowersForYou planner on the **10,000-example expanded dataset** (`data/train_expanded.jsonl`).

> [!CAUTION]
> **NO TRAINING WAS STARTED.** All existing adapter weights (`outputs/flowersforyou-adapter` and `outputs/phase2-adapter-backup`) remain strictly untouched. This report assesses configuration requirements so training can be executed safely once explicit user approval is granted.

---

## 1. Inspection of Current `train.py` Defaults

An inspection of `train.py` revealed several hardcoded configurations that must be addressed before training on the expanded dataset:

| Setting in `train.py` | Current Default | Expanded Dataset Risk / Impact | Recommended Adjustment |
|---|---|---|---|
| **Training File Path** | `ROOT / "data/train.jsonl"` | Points to 36-example baseline; will not train on expanded dataset | Add `--data-file` argument defaulting to `data/train_expanded.jsonl` |
| **Validation Dataset** | `ROOT / "data/eval.jsonl"` | **High Risk:** Contaminates the 9-example held-out test set by using it as a training evaluation split | Split an internal validation set (e.g. 200 records) from `data/train_expanded.jsonl` |
| **Output Adapter Dir** | `outputs/flowersforyou-adapter` | **Destructive Risk:** Overwrites the current active adapter weights | Direct outputs to a fresh directory: `outputs/flowersforyou-adapter-10k` |
| **Epochs (`num_train_epochs`)** | `3` | On 10,000 examples, 3 epochs = 30,000 sample exposures; causes severe overfitting on synthetic templates | Set `num_train_epochs=1` |
| **Batch Size & Grad Accum** | `batch_size=1`, `grad_accum=2` | Effective batch size of 2 $\to$ 15,000 optimizer steps (~5 hours on Apple MPS) | Increase to `batch_size=4`, `grad_accum=4` (effective batch size 16 $\to$ 625 steps, ~16 mins) |
| **Logging Frequency** | `logging_steps=1` | Emits 15,000 log lines to terminal and disk | Set `logging_steps=10` (emits 62 updates) |
| **Max Sequence Length** | `max_length=512` | Safe, but larger than necessary; mean token length is only 136 tokens | Keep `512` or set `384` for optimal MPS memory throughput |

---

## 2. Sequence Length & Truncation Empirical Audit

An empirical audit of all 10,000 records in `data/train_expanded.jsonl` was conducted using the Qwen tokenizer (`Qwen/Qwen2.5-0.5B-Instruct`):

- **Total records measured:** 10,000
- **Minimum token length:** 71 tokens
- **Maximum token length:** 265 tokens
- **Mean token length:** 136.5 tokens
- **Records exceeding 512 tokens:** **0 (0.0%)**
- **Records exceeding 384 tokens:** **0 (0.0%)**
- **Records exceeding 256 tokens:** 8 (0.08%)

### Verdict:
At `max_length=512`, **zero tokens and zero conversational turns are truncated** across the entire 10,000-example dataset. Even at `max_length=384`, zero truncation occurs.

---

## 3. Apple Silicon (MPS) Hardware & Memory Analysis

The user environment is a macOS Apple Silicon machine with PyTorch MPS acceleration enabled.

### Model & Activation Memory Footprint:
- **Base Model (`Qwen2.5-0.5B-Instruct` in `bfloat16`):** ~980 MB
- **LoRA Trainable Adapters ($r=8, \alpha=16$ on 7 projection matrices):** ~17.6 MB
- **Optimizer States (AdamW 32-bit for LoRA parameters):** ~70 MB
- **Activation Cache per Batch Item ($L=265$ tokens):** ~35 MB
- **Peak Unified Memory Usage (Batch Size 4, MPS):** **~2.2 GB - 2.8 GB**

### Verdict:
The 10,000-example fine-tuning run fits comfortably within local Apple Silicon unified memory (even on 8 GB or 16 GB unified RAM machines), with zero swap paging to disk.

---

## 4. Hyperparameter Calibration & Runtime Projections

### Option A: Current `train.py` Defaults (NOT RECOMMENDED)
- Batch Size: 1, Gradient Accumulation: 2 (Effective Batch Size: 2)
- Epochs: 3
- Total Optimizer Updates: $\frac{10,000 \times 3}{2} = \mathbf{15,000\text{ steps}}$
- Runtime @ 1.2s/step: **~5.0 hours**
- *Drawbacks:* High risk of overfitting to repetitive synthetic sentence starters; excessive logging (15,000 lines); unnecessarily long compute duration.

### Option B: Calibrated High-Efficiency SFT (RECOMMENDED)
- Batch Size: 4, Gradient Accumulation: 4 (Effective Batch Size: 16)
- Epochs: 1
- Total Optimizer Updates: $\frac{10,000 \times 1}{16} = \mathbf{625\text{ steps}}$
- Learning Rate: $1.2 \times 10^{-4}$ with linear warmup (50 steps)
- Regularization: `lora_dropout=0.1`
- Runtime @ 1.5s/step: **~15.6 minutes**
- *Benefits:* 10,000 diverse gradient exposures are sufficient for the 0.5B model to internalize safety refusals and canonical JSON formatting without memorizing specific phrasing stems; fast, low thermal load on Apple Silicon.

---

## 5. Train/Validation Split & Test Benchmark Isolation

To preserve rigorous evaluation standards:
1. **Never use `data/eval.jsonl` (9 cases) or `data/eval_expanded.jsonl` (56 cases) as the trainer's `eval_dataset` during SFT.**
2. When training on `data/train_expanded.jsonl`, carve out an internal validation split (e.g. 200 records / 2%) directly from `data/train_expanded.jsonl`.
3. Reserve `data/eval.jsonl` and `data/eval_expanded.jsonl` exclusively for post-training, independent zero-leakage evaluation.

---

## 6. Implementation Plan for Training Phase

When the user gives explicit approval to proceed with training, the recommended safe command will be:

```bash
# Recommended execution command once approved:
python train.py --train \
  --data-file data/train_expanded.jsonl \
  --output-dir outputs/flowersforyou-adapter-10k \
  --batch-size 4 \
  --grad-accum 4 \
  --epochs 1 \
  --learning-rate 1.2e-4
```

This ensures:
- The existing adapter in `outputs/flowersforyou-adapter` is preserved.
- The training runtime completes in approximately 15 minutes.
- The held-out evaluation sets remain untainted.
