"""Interference-phase fine-tune (no-replay protocol).

Continue LoRA fine-tuning on an interference set while periodically evaluating
learn-set probe accuracy. The IPR (Key Result 1) is the absolute drop in
flagged-probe accuracy — the gate metric that determines whether the replay
arms are worth running in Phase 2.

The no-replay interference loop is also the backbone for the replay arms:
Phase 2 will wrap this function inside a replay scheduler that interleaves
flagged learn-facts at a controlled token budget.

If no-replay fails to produce a ≥25-pt drop, the operator should tune
interference parameters (see docs/phase1-tuning.md) before escalating to
3B.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..data.facts import Fact
from ..data.probes import Probe
from ..evaluation.probe_eval import evaluate_probes
from .dataset import FactDataset, collate_padded

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class InterferenceCheckpoint:
    step: int
    flagged_acc: float
    unflagged_acc: float


@dataclass
class InterferencePhaseResult:
    steps_run: int
    initial_flagged_acc: float
    final_flagged_acc: float
    initial_unflagged_acc: float
    final_unflagged_acc: float
    checkpoints: list[InterferenceCheckpoint]
    train_loss_per_step: list[float]
    seconds: float

    def flagged_drop(self) -> float:
        """Absolute drop in flagged-probe accuracy — the gate metric."""
        return self.initial_flagged_acc - self.final_flagged_acc


def train_interference_phase(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    interference_facts: list[Fact],
    learn_facts: list[Fact],
    learn_probes: list[Probe],
    *,
    total_steps: int,
    lr: float,
    batch_size: int,
    max_seq_len: int,
    checkpoint_every: int,
    seed: int | None = None,
    save_dir: str | Path | None = None,
    device: str | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> InterferencePhaseResult:
    """Continue training on the interference set; checkpoint-eval probes.

    If *seed* is provided, the DataLoader uses a seeded ``torch.Generator``
    that advances each epoch, producing different shuffles across epochs
    while remaining reproducible across runs.

    If *save_dir* is provided, LoRA adapter weights are saved to
    ``<save_dir>/checkpoint_step_{step}.safetensors`` every ``checkpoint_every``
    steps so that partial progress survives a timeout or crash.
    """
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Baseline eval before any interference step — this is the value the
    # final accuracy is compared against for the ≥25-pt drop gate.
    baseline = evaluate_probes(model, tokenizer, learn_facts, learn_probes, device=device)
    model.train()

    dataset = FactDataset(interference_facts, tokenizer, max_length=max_seq_len)

    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    losses: list[float] = []
    checkpoints: list[InterferenceCheckpoint] = []
    checkpoints.append(
        InterferenceCheckpoint(
            step=0,
            flagged_acc=baseline["flagged_acc"],
            unflagged_acc=baseline["unflagged_acc"],
        )
    )

    t_start = time.time()
    last_eval = baseline

    # Per-epoch reshuffle: each epoch creates a fresh DataLoader with an
    # advancing seed (seed+0, seed+1, …). This gives different batch orders
    # across epochs while remaining reproducible across runs.  This is
    # preferred over itertools.cycle, which repeats the exact same shuffled
    # order every epoch.
    steps_done = 0
    epoch = 0
    while steps_done < total_steps:
        epoch_gen: torch.Generator | None = None
        if seed is not None:
            epoch_gen = torch.Generator(device="cpu").manual_seed(seed + epoch)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=epoch_gen,
            collate_fn=lambda b: collate_padded(b, pad_token_id=pad_id),
        )

        for batch in loader:
            if steps_done >= total_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(float(loss.item()))
            steps_done += 1
            step_num = steps_done  # 1-indexed step number

            if on_step is not None:
                on_step(step_num, float(loss.item()))

            if step_num % checkpoint_every == 0 or step_num == total_steps:
                eval_result = evaluate_probes(
                    model, tokenizer, learn_facts, learn_probes, device=device
                )
                checkpoints.append(
                    InterferenceCheckpoint(
                        step=step_num,
                        flagged_acc=eval_result["flagged_acc"],
                        unflagged_acc=eval_result["unflagged_acc"],
                    )
                )
                last_eval = eval_result
                model.train()

                # Persist LoRA adapter weights so partial progress survives
                # a Modal timeout or crash.
                if save_dir is not None:
                    save_path = Path(save_dir) / f"checkpoint_step_{step_num:06d}"
                    save_path.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(save_path)

        epoch += 1

    return InterferencePhaseResult(
        steps_run=total_steps,
        initial_flagged_acc=baseline["flagged_acc"],
        final_flagged_acc=last_eval["flagged_acc"],
        initial_unflagged_acc=baseline["unflagged_acc"],
        final_unflagged_acc=last_eval["unflagged_acc"],
        checkpoints=checkpoints,
        train_loss_per_step=losses,
        seconds=time.time() - t_start,
    )
