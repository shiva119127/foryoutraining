#!/usr/bin/env python3
"""
FlowersForYou - Model Inference.

Loads the base model (Qwen/Qwen2.5-0.5B-Instruct) and optionally attaches the
trained PEFT LoRA adapter from outputs/flowersforyou-adapter.
Formats prompts using the native Jinja2 chat template.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Enable MPS fallback for any unsupported operations on Apple Silicon
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ADAPTER_DIR = ROOT / "outputs/flowersforyou-adapter"

SYSTEM_PROMPTS = {
    "planner": (
        "You are the FlowersForYou planner. Be helpful, concise, and warm. "
        "Never invent real-time venue availability, live pricing, or confirmed bookings. "
        "Ask essential clarifying questions to narrow down options."
    ),
    "onboarding": (
        "You are the FlowersForYou onboarding assistant. Learn preferences naturally. "
        "Ask one relevant question at a time. Do not turn onboarding into a rigid survey."
    ),
    "returning": (
        "You are the FlowersForYou assistant. If a returning user has completed onboarding, "
        "do not repeat onboarding. Greet them warmly and invite a new plan."
    ),
}

DEFAULT_PROMPTS = {
    "planner": "Hi! I'm looking for a board-game meetup in Koramangala this weekend. My budget is around ₹500.",
    "onboarding": "I like meeting new people, but big noisy crowds make me uncomfortable.",
    "returning": "Hey, I'm back. Can you help me plan something fun?",
}


def get_device_and_dtype():
    """Select device and dtype suitable for local Apple Silicon or CPU fallback."""
    import torch

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.bfloat16, "Apple Silicon MPS (bfloat16)"
    return torch.device("cpu"), torch.float32, "CPU fallback (float32)"


def main():
    parser = argparse.ArgumentParser(description="FlowersForYou Model Inference (Base or Adapter).")
    parser.add_argument("--prompt", type=str, default=None, help="User prompt to generate a response for.")
    parser.add_argument(
        "--role",
        choices=["planner", "onboarding", "returning"],
        default="planner",
        help="System role/phase persona (planner, onboarding, returning).",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=str(DEFAULT_ADAPTER_DIR),
        help="Path to trained PEFT adapter directory.",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Force inference with the base model without loading the adapter.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=80,
        help="Maximum new tokens to generate (default: 80).",
    )
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    adapter_config_file = adapter_path / "adapter_config.json"
    use_adapter = not args.base_only

    if use_adapter and not adapter_config_file.exists():
        print("=" * 60)
        print("FlowersForYou Inference Notice")
        print("=" * 60)
        print(f"[INFO] Adapter not found at: {adapter_path}")
        print("Training has not been run yet to generate this adapter.")
        print("\nOptions:")
        print("  1. Train the adapter:        python train.py --train")
        print("  2. Test with base model:     python inference.py --base-only")
        print("  3. Run Phase 1 load test:    python test_model_load.py")
        print("=" * 60)
        sys.exit(0)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if use_adapter:
            from peft import PeftModel
    except ImportError as exc:
        print(f"\n[FAIL] Missing required dependencies: {exc}")
        sys.exit(1)

    device, dtype, env_desc = get_device_and_dtype()
    mode_label = f"Fine-Tuned Adapter ({adapter_path.name})" if use_adapter else "Base Model Only"

    print("=" * 60)
    print("FlowersForYou Inference")
    print("=" * 60)
    print(f"Base Model:    {MODEL_ID}")
    print(f"Execution:     {mode_label}")
    print(f"Environment:   {env_desc}")
    print(f"Persona Role:  {args.role}")

    # 1. Load Tokenizer
    print(f"\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as exc:
        print(f"\n[FAIL] Could not load tokenizer: {exc}")
        sys.exit(1)

    # 2. Load Base Model
    print(f"Loading base model (dtype={dtype})...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
        )
        base_model = base_model.to(device)
    except Exception as exc:
        if device.type == "mps":
            print(f"[WARN] Failed to load on MPS ({exc}). Falling back to CPU...")
            device = torch.device("cpu")
            dtype = torch.float32
            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
            )
            base_model = base_model.to(device)
        else:
            print(f"\n[FAIL] Failed to load base model: {exc}")
            sys.exit(1)

    # 3. Load Adapter if requested
    if use_adapter:
        print(f"Attaching PEFT adapter from {adapter_path}...")
        try:
            model = PeftModel.from_pretrained(base_model, str(adapter_path))
            model = model.to(device)
        except Exception as exc:
            print(f"\n[FAIL] Failed to attach adapter: {exc}")
            sys.exit(1)
    else:
        model = base_model

    model.eval()

    # 4. Prepare prompt & apply chat template
    user_prompt = args.prompt or DEFAULT_PROMPTS[args.role]
    system_prompt = SYSTEM_PROMPTS[args.role]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        print(f"\n[FAIL] Failed to apply chat template: {exc}")
        sys.exit(1)

    print("\n" + "-" * 60)
    print(f"User Prompt: \"{user_prompt}\"")

    # 5. Generate completion
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    start_time = time.time()
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - start_time
    except Exception as exc:
        print(f"\n[FAIL] Generation failed: {exc}")
        sys.exit(1)

    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print("\nGenerated Response:")
    print("=" * 60)
    print(response)
    print("=" * 60)
    tps = len(new_tokens) / elapsed if elapsed > 0 else 0
    print(f"Tokens: {len(new_tokens)} | Time: {elapsed:.2f}s ({tps:.1f} tokens/s)\n")


if __name__ == "__main__":
    main()
