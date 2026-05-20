"""Modal app for dreamcue.

GPU: Modal H100 (NVIDIA, CUDA). The PRD originally targeted an AMD MI300X
endpoint but Modal's public GPU lineup does not include MI300X, so the
operator confirmed on 2026-05-19 to run on Modal H100 and drop the AMD
framing from the reel. See docs/decisions.md.

The code still supports a ROCm path for completeness (DREAMCUE_GPU=MI*
swaps to a rocm/pytorch base image) in case the operator gets MI300X
access later. Default is H100/CUDA.
"""

from __future__ import annotations

import os

import modal

DREAMCUE_GPU = os.environ.get("DREAMCUE_GPU", "H100")
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

_HERE = os.path.dirname(os.path.abspath(__file__))

# Mount the local package source and configs into the container so we can
# import `dreamcue.*` and read configs/default.yaml without publishing a wheel.
# `copy=False` (default) ships the files at container startup — fast iteration.
image = image.add_local_dir(os.path.join(_HERE, "src"), remote_path="/root/src")
image = image.add_local_dir(os.path.join(_HERE, "configs"), remote_path="/root/configs")

app = modal.App("dreamcue", image=image)

# Volume for caching HF model weights between runs so we don't re-download 2.4GB
# every Modal cold start.
hf_cache = modal.Volume.from_name("dreamcue-hf-cache", create_if_missing=True)

# Volume for persisting Phase 1 / 2 artifacts (summaries, loss curves).
results_volume = modal.Volume.from_name("dreamcue-results", create_if_missing=True)

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


@app.function(
    gpu=DREAMCUE_GPU,
    timeout=TIMEOUT_S * 4,  # Phase 1 is longer than the smoke test
    secrets=[hf_secret],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/results": results_volume,
    },
)
def phase1_gate(
    config_path: str = "/root/configs/default.yaml",
    seed: int | None = None,
    tiny: bool = False,
) -> dict:
    """Phase 1: learn-phase → no-replay interference → measure forgetting.

    Returns:
        dict with initial/final flagged accuracy, drop, gate_passed (bool),
        and paths to artifacts on the results volume.

    Args:
        config_path: YAML config, defaults to the mounted configs/default.yaml.
        seed: override config seed (used by Phase 2 to run multiple seeds).
        tiny: shrink dataset + steps for a 1-2 minute end-to-end smoke run.
            Use this before launching the full sweep — proves the wiring on
            the actual GPU without burning the full budget.
    """
    import json
    import os
    import sys
    import time

    sys.path.insert(0, "/root/src")

    import torch
    import yaml
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dreamcue.data.facts import build_learn_set
    from dreamcue.data.interference import build_interference_set
    from dreamcue.data.probes import build_probes
    from dreamcue.training.learn_loop import train_learn_phase
    from dreamcue.training.interference_loop import train_interference_phase

    cfg = yaml.safe_load(open(config_path).read())

    data_cfg = dict(cfg["data"])
    learn_cfg = dict(cfg["learn_phase"])
    inter_cfg = dict(cfg["interference_phase"])

    if tiny:
        data_cfg["n_learn_facts"] = 50
        data_cfg["n_interference_facts"] = 200
        learn_cfg["epochs"] = 3
        inter_cfg["total_steps"] = 100
        inter_cfg["checkpoint_every"] = 50

    if seed is not None:
        data_cfg["seed"] = seed

    info: dict = {
        "config_path": config_path,
        "tiny": tiny,
        "seed": data_cfg["seed"],
        "model": cfg["model"]["name"],
    }
    t_total = time.time()

    learn_facts = build_learn_set(
        n_facts=data_cfg["n_learn_facts"],
        flag_rate=data_cfg["flag_rate"],
        seed=data_cfg["seed"],
    )
    probes = build_probes(
        learn_facts,
        paraphrases_per_fact=data_cfg["probe_paraphrases_per_fact"],
        seed=data_cfg["seed"],
    )
    interference_facts = build_interference_set(
        learn_facts,
        n_facts=data_cfg["n_interference_facts"],
        seed=data_cfg["seed"] + 1,
    )
    info["n_learn"] = len(learn_facts)
    info["n_probes"] = len(probes)
    info["n_interference"] = len(interference_facts)

    # Model + tokenizer.
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], token=os.environ["HF_TOKEN"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left-pad for generation
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        torch_dtype=torch.float16,
        token=os.environ["HF_TOKEN"],
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.to("cuda")
    info["model_load_seconds"] = round(time.time() - t0, 2)

    # Learn phase.
    learn_result = train_learn_phase(
        model=model,
        tokenizer=tok,
        facts=learn_facts,
        probes=probes,
        epochs=learn_cfg["epochs"],
        lr=learn_cfg["lr"],
        batch_size=learn_cfg["batch_size"],
        max_seq_len=learn_cfg["max_seq_len"],
        target_probe_accuracy=learn_cfg["target_probe_accuracy"],
    )
    info["learn"] = {
        "epochs_run": learn_result.epochs_run,
        "final_flagged_acc": learn_result.final_flagged_acc,
        "final_unflagged_acc": learn_result.final_unflagged_acc,
        "seconds": round(learn_result.seconds, 1),
        "eval_history": learn_result.eval_history,
    }

    # Interference phase (no replay — this is the forgetting-floor control).
    inter_result = train_interference_phase(
        model=model,
        tokenizer=tok,
        interference_facts=interference_facts,
        learn_facts=learn_facts,
        learn_probes=probes,
        total_steps=inter_cfg["total_steps"],
        lr=inter_cfg["lr"],
        batch_size=inter_cfg["batch_size"],
        max_seq_len=learn_cfg["max_seq_len"],
        checkpoint_every=inter_cfg["checkpoint_every"],
    )
    drop = inter_result.flagged_drop() * 100  # absolute percentage points
    info["interference"] = {
        "steps_run": inter_result.steps_run,
        "initial_flagged_acc": inter_result.initial_flagged_acc,
        "final_flagged_acc": inter_result.final_flagged_acc,
        "initial_unflagged_acc": inter_result.initial_unflagged_acc,
        "final_unflagged_acc": inter_result.final_unflagged_acc,
        "flagged_drop_pts": round(drop, 2),
        "seconds": round(inter_result.seconds, 1),
        "checkpoints": [
            {"step": c.step, "flagged_acc": c.flagged_acc, "unflagged_acc": c.unflagged_acc}
            for c in inter_result.checkpoints
        ],
    }

    info["gate_passed"] = drop >= 25.0
    info["gate_metric"] = "flagged_probe_accuracy_drop_pts >= 25"
    info["total_seconds"] = round(time.time() - t_total, 1)

    # Persist artifacts.
    run_tag = f"phase1-seed{data_cfg['seed']}{'-tiny' if tiny else ''}"
    out_dir = f"/root/results/{run_tag}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(info, f, indent=2)
    with open(f"{out_dir}/learn_loss.csv", "w") as f:
        f.write("step,loss\n")
        for i, loss in enumerate(learn_result.train_loss_per_step):
            f.write(f"{i},{loss}\n")
    with open(f"{out_dir}/interference_loss.csv", "w") as f:
        f.write("step,loss\n")
        for i, loss in enumerate(inter_result.train_loss_per_step):
            f.write(f"{i + 1},{loss}\n")
    results_volume.commit()
    info["artifacts_dir"] = out_dir

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


@app.local_entrypoint()
def phase1(tiny: bool = False, seed: int | None = None):
    """Run Phase 1: learn → no-replay interference → measure forgetting.

    Usage:
        modal run modal_app.py::phase1 --tiny        # 1-2 min wiring check
        modal run modal_app.py::phase1               # full run on seed 1337
        modal run modal_app.py::phase1 --seed 22     # alt seed
    """
    result = phase1_gate.remote(tiny=tiny, seed=seed)
    print("=== Phase 1 gate ===")
    print(f"  model:                {result['model']}")
    print(f"  seed:                 {result['seed']}  tiny={result['tiny']}")
    print(f"  n_learn / n_probes:   {result['n_learn']} / {result['n_probes']}")
    print(f"  n_interference:       {result['n_interference']}")
    learn = result["learn"]
    print(f"  learn epochs:         {learn['epochs_run']}  "
          f"flagged_acc={learn['final_flagged_acc']:.3f}  "
          f"unflagged_acc={learn['final_unflagged_acc']:.3f}")
    inter = result["interference"]
    print(f"  interference steps:   {inter['steps_run']}")
    print(f"  flagged acc:          {inter['initial_flagged_acc']:.3f} → "
          f"{inter['final_flagged_acc']:.3f}  (drop {inter['flagged_drop_pts']:.1f} pts)")
    print(f"  unflagged acc:        {inter['initial_unflagged_acc']:.3f} → "
          f"{inter['final_unflagged_acc']:.3f}")
    print(f"  GATE ({result['gate_metric']}): "
          f"{'PASS' if result['gate_passed'] else 'FAIL'}")
    print(f"  artifacts:            {result['artifacts_dir']}")
    print(f"  total seconds:        {result['total_seconds']}")
