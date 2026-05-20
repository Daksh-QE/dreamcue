"""Modal app for dreamcue.

Per PRD §5.1 and §6 Phase 0, the target is an AMD MI300X endpoint via Modal.
NOTE: as of this writing, Modal's publicly documented GPU types are NVIDIA-only
(T4, L4, A10, L40S, A100, RTX-PRO-6000, H100, H200, B200). If MI300X is not
available on the user's Modal workspace, the Phase 0 smoke test will fail at
function attachment with a clear error from Modal — at which point we surface
the fallback choice (private AMD endpoint, or H100 + a framing note for the
reel) to the operator before any spend.

Configure via env var DREAMCUE_GPU (default "MI300X"). The image picks a
ROCm PyTorch base when DREAMCUE_GPU starts with "MI" (AMD), otherwise a CUDA
PyTorch wheel. This keeps the same code path for both backends.
"""

from __future__ import annotations

import os

import modal

DREAMCUE_GPU = os.environ.get("DREAMCUE_GPU", "MI300X")
IS_AMD = DREAMCUE_GPU.upper().startswith("MI")

MODEL_NAME = os.environ.get("DREAMCUE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
TIMEOUT_S = int(os.environ.get("DREAMCUE_TIMEOUT_S", "1800"))

# Image: AMD path uses a ROCm PyTorch base; NVIDIA path uses the standard CUDA wheels.
# Both install the same downstream Python deps so call sites are identical.
COMMON_PIP = [
    "transformers==4.46.3",
    "peft==0.14.0",
    "accelerate==1.2.1",
    "datasets==3.2.0",
    "huggingface-hub==0.27.0",
    "numpy>=1.26",
    "pandas>=2.2",
    "scipy>=1.13",
    "pyyaml>=6.0",
    "tqdm>=4.66",
]

if IS_AMD:
    # ROCm PyTorch image. The rocm/pytorch tags ship with a matched PyTorch build.
    # Pinned to a known-good ROCm 6.2 / PyTorch 2.4 combo; bump as ROCm releases.
    image = (
        modal.Image.from_registry(
            "rocm/pytorch:rocm6.2.4_ubuntu22.04_py3.10_pytorch_release_2.4.0",
            add_python="3.10",
        )
        .pip_install(*COMMON_PIP)
        .env({"DREAMCUE_BACKEND": "rocm"})
    )
else:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.4 --index-url https://download.pytorch.org/whl/cu121",
            *COMMON_PIP,
        )
        .env({"DREAMCUE_BACKEND": "cuda"})
    )

app = modal.App("dreamcue", image=image)

# Volume for caching HF model weights between runs so we don't re-download 2.4GB
# every Modal cold start.
hf_cache = modal.Volume.from_name("dreamcue-hf-cache", create_if_missing=True)

# HuggingFace token comes from a Modal Secret named "huggingface".
# Create it once locally with:  modal secret create huggingface HF_TOKEN=hf_xxx
hf_secret = modal.Secret.from_name("huggingface")


def _gpu_telemetry() -> str:
    """Best-effort GPU telemetry capture for reel B-roll.

    Tries rocm-smi (AMD) first, then nvidia-smi (NVIDIA). Returns the raw
    output as a string; empty on failure.
    """
    import shutil
    import subprocess

    for binary, args in [
        ("rocm-smi", ["--showuse", "--showmemuse", "--showtemp", "--showpower"]),
        ("amd-smi", ["monitor", "-u", "-m", "-p", "-t"]),
        ("nvidia-smi", []),
    ]:
        if shutil.which(binary):
            try:
                out = subprocess.run(
                    [binary, *args], capture_output=True, text=True, timeout=10
                )
                return f"$ {binary} {' '.join(args)}\n{out.stdout}\n{out.stderr}"
            except subprocess.SubprocessError:
                continue
    return "(no GPU telemetry binary found in container)"


@app.function(
    gpu=DREAMCUE_GPU,
    timeout=TIMEOUT_S,
    secrets=[hf_secret],
    volumes={"/root/.cache/huggingface": hf_cache},
)
def smoke_test(dry_run: bool = False) -> dict:
    """Phase 0 smoke test: load Llama-3.2-1B-Instruct, run one LoRA step on dummy data.

    Returns a dict with backend info, loss value, and telemetry. Failure modes:
    - Modal can't allocate the requested GPU → fails before we enter this function.
    - HF token missing or model gated → raises in from_pretrained.
    - LoRA step fails → raises with traceback.

    If dry_run is True, skips the model load and just reports the backend.
    """
    import os
    import time

    backend = os.environ.get("DREAMCUE_BACKEND", "unknown")
    info = {"backend": backend, "gpu_requested": DREAMCUE_GPU, "model": MODEL_NAME}

    if dry_run:
        info["dry_run"] = True
        info["telemetry"] = _gpu_telemetry()
        return info

    import torch

    info["torch_version"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if info["device_count"]:
        info["device_name"] = torch.cuda.get_device_name(0)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ["HF_TOKEN"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        token=os.environ["HF_TOKEN"],
    )
    info["load_seconds"] = round(time.time() - t0, 2)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.to("cuda")  # ROCm exposes itself as torch.cuda
    model.train()

    # One step on dummy data.
    prompt = "Entity ABC works at WidgetCo. Entity ABC works at"
    target = " WidgetCo."
    enc = tok(prompt + target, return_tensors="pt", padding=True).to("cuda")
    labels = enc["input_ids"].clone()
    out = model(**enc, labels=labels)
    out.loss.backward()
    info["smoke_loss"] = float(out.loss.item())

    # Telemetry snapshot.
    info["telemetry"] = _gpu_telemetry()

    # Save + reload roundtrip — proves checkpoint integrity before any real run.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, token=os.environ["HF_TOKEN"]
        )
        _reloaded = PeftModel.from_pretrained(base, tmp)
        info["checkpoint_roundtrip"] = "ok"

    return info


@app.local_entrypoint()
def main(dry_run: bool = False):
    """Run the smoke test from your laptop:  modal run modal_app.py --dry-run"""
    result = smoke_test.remote(dry_run=dry_run)
    print("=== dreamcue smoke test ===")
    for k, v in result.items():
        if k == "telemetry":
            print(f"\n[telemetry]\n{v}")
        else:
            print(f"  {k}: {v}")
