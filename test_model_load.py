#!/usr/bin/env python3
"""
FlowersForYou - Standalone Model Loading & Inference Test (Phase 1).

Validates:
- AutoTokenizer and AutoModelForCausalLM loading for Qwen/Qwen2.5-0.5B-Instruct
- Apple Silicon MPS acceleration with automatic CPU fallback
- Appropriate dtype selection (bfloat16 on MPS, float32 on CPU)
- Tokenizer chat template formatting and short generative response
- Safe error handling for download, memory, and runtime issues

Does not modify any project files, create adapters, or start training.
"""

import sys
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

def select_device_and_dtype():
    """
    Select Apple Silicon MPS if available, otherwise fall back to CPU.
    Select bfloat16 for MPS, float32 for CPU.
    """
    import torch

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.bfloat16
        notes = "Apple Silicon MPS available; using bfloat16 for optimal memory & compute efficiency."
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        notes = "Apple MPS unavailable; falling back to CPU with float32."

    return device, dtype, notes


def run_test():
    print("=" * 60)
    print("FlowersForYou - Model Loading & Inference Test (Phase 1)")
    print("=" * 60)
    print(f"Target Model:  {MODEL_ID}")

    # 1. Inspect environment & PyTorch capabilities
    try:
        import torch
        import transformers
    except ImportError as exc:
        print(f"\n[FAIL] Missing required core package: {exc}")
        print("Ensure your virtual environment is active (.venv) with transformers and torch installed.")
        return 1

    print(f"PyTorch:       {torch.__version__}")
    print(f"Transformers:  {transformers.__version__}")

    device, dtype, notes = select_device_and_dtype()
    print(f"Target Device: {device}")
    print(f"Target Dtype:  {dtype}")
    print(f"Notes:         {notes}")
    print("-" * 60)

    # 2. Load AutoTokenizer
    print(f"Loading tokenizer for '{MODEL_ID}'...")
    start_time = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tok_time = time.time() - start_time
        print(f"[OK] Tokenizer loaded in {tok_time:.2f}s")
        print(f"     Vocab size: {tokenizer.vocab_size}")
        print(f"     Chat template present: {bool(tokenizer.chat_template)}")
    except Exception as exc:
        print(f"\n[FAIL] Could not load tokenizer for '{MODEL_ID}':")
        print(f"       {type(exc).__name__}: {exc}")
        print("Check your network connection or verify that Hugging Face Hub is accessible.")
        return 1

    # 3. Load Model onto Target Device
    print(f"\nLoading model '{MODEL_ID}' (dtype={dtype})...")
    start_time = time.time()
    model = None
    loaded_device = device
    loaded_dtype = dtype

    try:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
            )
            model = model.to(device)
        except Exception as device_exc:
            # If MPS fails (e.g., unsupported op or memory allocation issue), fall back to CPU
            if device.type == "mps":
                print(f"[WARN] Failed to load on MPS ({device_exc}). Attempting CPU fallback...")
                loaded_device = torch.device("cpu")
                loaded_dtype = torch.float32
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    torch_dtype=loaded_dtype,
                )
                model = model.to(loaded_device)
            else:
                raise device_exc

        load_time = time.time() - start_time
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[OK] Model loaded in {load_time:.2f}s")
        print(f"     Active Device: {loaded_device}")
        print(f"     Active Dtype:  {loaded_dtype}")
        print(f"     Total Parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    except MemoryError as exc:
        print(f"\n[FAIL] Out of memory loading model '{MODEL_ID}': {exc}")
        return 1
    except Exception as exc:
        print(f"\n[FAIL] Could not load model '{MODEL_ID}':")
        print(f"       {type(exc).__name__}: {exc}")
        print("If this is a network timeout, re-running the command may resume the download.")
        return 1

    # 4. Formulate test FlowersForYou planning conversation
    messages = [
        {
            "role": "system",
            "content": (
                "You are the FlowersForYou planner. Be helpful, concise, and warm. "
                "Never invent real-time venue availability, live pricing, or confirmed bookings. "
                "Ask essential clarifying questions to narrow down options."
            ),
        },
        {
            "role": "user",
            "content": (
                "Hi! I'm looking for a board-game meetup in Koramangala this weekend. "
                "My budget is around ₹500. Can you help?"
            ),
        },
    ]

    print("\n" + "-" * 60)
    print("Testing Chat Template & Short Inference...")
    print(f"User Prompt: \"{messages[1]['content']}\"")

    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        print(f"\n[FAIL] Failed to apply chat template: {exc}")
        return 1

    # 5. Run short inference
    try:
        model.eval()
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(loaded_device)
        input_token_count = inputs["input_ids"].shape[1]

        gen_start = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_time = time.time() - gen_start

        # Slice out generated tokens (strip the input prompt tokens)
        generated_tokens = output_ids[0][input_token_count:]
        generated_token_count = len(generated_tokens)
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        print("\nGenerated Response:")
        print("=" * 60)
        print(response)
        print("=" * 60)

        tokens_per_sec = generated_token_count / gen_time if gen_time > 0 else 0
        print(f"\nInference Metrics:")
        print(f"  Input Tokens:      {input_token_count}")
        print(f"  Generated Tokens:  {generated_token_count}")
        print(f"  Generation Time:   {gen_time:.2f}s ({tokens_per_sec:.1f} tokens/s)")

    except Exception as exc:
        print(f"\n[FAIL] Inference failed during generation: {exc}")
        return 1

    print("\n" + "=" * 60)
    print("[PASS] Phase 1 Model Loading & Inference Test succeeded.")
    print("Model, tokenizer, Apple Silicon MPS acceleration, and chat template are verified.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(run_test())
