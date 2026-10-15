import argparse
import importlib.metadata
import os
import sys
from pathlib import Path

# Enable MPS fallback for any unsupported operations on Apple Silicon
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Apply Python 3.14 compatibility patch for datasets dill pickler
if sys.version_info >= (3, 14):
    try:
        import datasets.utils._dill as _datasets_dill
        import dill

        def _fixed_batch_setitems(self, items, obj=None):
            if getattr(self, "_legacy_no_dict_keys_sorting", False):
                if obj is not None:
                    return super(_datasets_dill.Pickler, self)._batch_setitems(items, obj)
                return super(_datasets_dill.Pickler, self)._batch_setitems(items)
            try:
                items = sorted(items)
            except Exception:
                from datasets.fingerprint import Hasher

                items = sorted(items, key=lambda x: Hasher.hash(x[0]))
            if obj is not None:
                dill.Pickler._batch_setitems(self, items, obj)
            else:
                dill.Pickler._batch_setitems(self, items)

        _datasets_dill.Pickler._batch_setitems = _fixed_batch_setitems
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def get_device_and_dtype():
    """Select device and dtype suitable for local Apple Silicon or CPU fallback."""
    import torch

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.bfloat16, True
    return torch.device("cpu"), torch.float32, False


def preflight():
    print("=" * 60)
    print("FlowersForYou Training Preflight")
    print("=" * 60)
    print("Candidate model:", MODEL_ID)
    print("Dataset files:  ", ROOT / "data/train.jsonl", "and", ROOT / "data/eval.jsonl")
    for package in ("torch", "transformers", "datasets", "trl", "peft", "accelerate"):
        try:
            print(f"  {package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"  {package}: MISSING")
            return False
    try:
        import torch

        print("PyTorch version:", torch.__version__)
        print("CUDA available: ", torch.cuda.is_available())
        if hasattr(torch.backends, "mps"):
            print("Apple MPS available:", torch.backends.mps.is_available())
    except Exception as exc:
        print("PyTorch preflight failed:", repr(exc))
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="FlowersForYou SFT training (explicit opt-in required).")
    parser.add_argument("--train", action="store_true", help="Explicitly start training after preflight.")
    args = parser.parse_args()

    if not preflight():
        print("\nPreflight failed. Fix environment issues first. No training started.")
        sys.exit(1)

    # Import the local validator and require clean data before any training.
    from validate_data import validate_file

    all_ok = True
    for name in ("train.jsonl", "eval.jsonl"):
        count, errors = validate_file(ROOT / "data" / name)
        print(f"{name}: {count} record(s)")
        if errors:
            all_ok = False
            for error in errors:
                print("  ERROR:", error)
    if not all_ok:
        print("\nDataset validation failed. No training started.")
        sys.exit(1)

    if not args.train:
        print("\nPRE-FLIGHT ONLY: checks passed. Training was NOT started.")
        print("Review model license, chat template, hardware and this script before using --train.")
        return

    print("\nExplicit --train flag supplied. Loading training stack...")
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:
        print("\n[FAIL] Could not import training dependencies:", repr(exc))
        sys.exit(1)

    device, dtype, use_mps = get_device_and_dtype()
    print(f"Target device: {device} | Dtype: {dtype} | MPS acceleration: {use_mps}")

    train_path = ROOT / "data/train.jsonl"
    eval_path = ROOT / "data/eval.jsonl"
    try:
        train_data = load_dataset("json", data_files=str(train_path), split="train")
        eval_data = load_dataset("json", data_files=str(eval_path), split="train")
        print(f"Loaded {len(train_data)} train records, {len(eval_data)} eval records.")
    except Exception as exc:
        print(f"\n[FAIL] Could not load datasets: {exc}")
        sys.exit(1)

    # 1. Load Tokenizer explicitly
    print(f"Loading tokenizer: {MODEL_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as exc:
        print(f"\n[FAIL] Failed to load tokenizer for '{MODEL_ID}': {exc}")
        sys.exit(1)

    # 2. Load Base Model explicitly with device & dtype handling
    print(f"Loading base model: {MODEL_ID} (dtype={dtype})...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
        )
        model = model.to(device)
    except Exception as exc:
        if use_mps:
            print(f"[WARN] Failed to load on MPS ({exc}). Falling back to CPU...")
            device = torch.device("cpu")
            dtype = torch.float32
            use_mps = False
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
            )
            model = model.to(device)
        else:
            print(f"\n[FAIL] Failed to load model '{MODEL_ID}': {exc}")
            sys.exit(1)

    # 3. Configure LoRA target modules explicitly for Qwen2 architecture
    # Qwen2 attention modules: q_proj, k_proj, v_proj, o_proj
    # Qwen2 MLP modules: gate_proj, up_proj, down_proj
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,  # 0.1 dropout for regularization against overfitting
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    output_dir = ROOT / "outputs/flowersforyou-adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Training hyperparameters calibrated for 30-example dataset:
    # batch_size=1, gradient_accumulation_steps=2 -> 15 optimizer steps per epoch.
    # num_train_epochs=3 -> 45 total optimizer updates with balanced gradient smoothing.
    config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1.5e-4,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        max_length=512,
        bf16=use_mps,
        fp16=False,
    )

    # 5. Initialize SFTTrainer with processing_class=tokenizer (TRL 0.26+ API)
    try:
        trainer = SFTTrainer(
            model=model,
            args=config,
            train_dataset=train_data,
            eval_dataset=eval_data,
            peft_config=lora_config,
            processing_class=tokenizer,
        )
    except Exception as exc:
        print(f"\n[FAIL] SFTTrainer initialization failed: {exc}")
        sys.exit(1)

    print("\nStarting training loop...")
    try:
        trainer.train()
        print(f"\nTraining completed successfully. Saving adapter to: {output_dir}")
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print("Adapter weights and tokenizer configurations saved.")
        print("Next steps:")
        print("  1. Run inference:  python inference.py")
        print("  2. Run evaluation: python evaluate.py")
    except Exception as exc:
        print(f"\n[FAIL] Training execution failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
