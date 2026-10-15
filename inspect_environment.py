import importlib.metadata
import platform
import shutil
import sys

print("FlowersForYou environment preflight")
print("=" * 40)
print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
print("Machine:", platform.machine())
print("Architecture:", platform.architecture()[0])
print("CUDA compiler/toolkit command:", shutil.which("nvidia-smi") or "nvidia-smi not found (normal on Mac)")
print("\nInstalled packages:")
for package in ("torch", "transformers", "datasets", "trl", "peft", "accelerate", "safetensors"):
    try:
        print(f"  {package}: {importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"  {package}: NOT INSTALLED")

try:
    import torch
    print("\nPyTorch:")
    print("  version:", torch.__version__)
    print("  CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  CUDA device:", torch.cuda.get_device_name(0))
    if hasattr(torch.backends, "mps"):
        print("  Apple Metal (MPS) available:", torch.backends.mps.is_available())
except Exception as exc:
    print("\nCould not import torch:", repr(exc))

print("\nThis is an environment report, not a guarantee that the selected model will fit.")
print("No model was downloaded and no training was started.")
