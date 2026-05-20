"""Interference-phase fine-tune (no replay).

This is the forgetting-floor control. The model that just learned the
learn-set now trains on the interference set only — no learn-fact replay.
We measure flagged-probe accuracy every `checkpoint_every` steps so the
retention curve can be drawn.

The Phase 1 gate requires the flagged-probe accuracy to drop by ≥25 absolute
points between the end of the learn phase and the end of the interference
phase under this no-replay protocol. If that doesn't happen, the experiment
is broken — there's no forgetting to recover from.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from ..data.facts import Fact
from ..data.probes import Probe
from ..evaluation.probe_eval import evaluate_probes
from .dataset import FactDataset, collate_padded

if TYPE_CHECKING:
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
    device: str | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> InterferencePhaseResult:
    """Continue training on the interference set; checkpoint-eval probes."""
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
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_padded(b, pad_token_id=pad_id),
    )

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
    step = 0
    last_eval = baseline

    iterator = iter(loader)
    while step < total_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(float(loss.item()))
        step += 1
        if on_step is not None:
            on_step(step, float(loss.item()))

        if step % checkpoint_every == 0 or step == total_steps:
            eval_result = evaluate_probes(
                model, tokenizer, learn_facts, learn_probes, device=device
            )
            checkpoints.append(
                InterferenceCheckpoint(
                    step=step,
                    flagged_acc=eval_result["flagged_acc"],
                    unflagged_acc=eval_result["unflagged_acc"],
                )
            )
            last_eval = eval_result
            model.train()

    return InterferencePhaseResult(
        steps_run=step,
        initial_flagged_acc=baseline["flagged_acc"],
        final_flagged_acc=last_eval["flagged_acc"],
        initial_unflagged_acc=baseline["unflagged_acc"],
        final_unflagged_acc=last_eval["unflagged_acc"],
        checkpoints=checkpoints,
        train_loss_per_step=losses,
        seconds=time.time() - t_start,
    )
