"""Learn-phase fine-tune.

Train LoRA on the entire learn-set until flagged-probe accuracy reaches
`target_probe_accuracy` (default 0.90) or `epochs` is exhausted. The output
is a PEFT model with adapters loaded — caller hands it to the interference
loop next.

Pure training utility — no Modal coupling. The Modal entrypoint imports
this module and runs `train_learn_phase` inside the container.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from ..data.facts import Fact
from ..data.probes import Probe
from ..evaluation.probe_eval import evaluate_probes
from .dataset import FactDataset, collate_padded

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class LearnPhaseResult:
    epochs_run: int
    final_flagged_acc: float
    final_unflagged_acc: float
    train_loss_per_step: list[float]
    eval_history: list[dict[str, float]]  # per-epoch [{epoch, flagged_acc, unflagged_acc}]
    seconds: float


def train_learn_phase(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    facts: list[Fact],
    probes: list[Probe],
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    max_seq_len: int,
    target_probe_accuracy: float,
    device: str | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> LearnPhaseResult:
    """Run the learn-phase fine-tune.

    Returns when flagged probe accuracy ≥ target_probe_accuracy or epoch
    budget is exhausted, whichever comes first.
    """
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    dataset = FactDataset(facts, tokenizer, max_length=max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_padded(b, pad_token_id=pad_id),
    )

    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    model.train()
    losses: list[float] = []
    eval_history: list[dict[str, float]] = []
    t_start = time.time()
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(float(loss.item()))
            if on_step is not None:
                on_step(step, float(loss.item()))

        # End-of-epoch eval.
        eval_result = evaluate_probes(model, tokenizer, facts, probes, device=device)
        eval_history.append(
            {
                "epoch": epoch,
                "flagged_acc": eval_result["flagged_acc"],
                "unflagged_acc": eval_result["unflagged_acc"],
            }
        )
        model.train()

        if eval_result["flagged_acc"] >= target_probe_accuracy:
            break

    return LearnPhaseResult(
        epochs_run=epochs_run,
        final_flagged_acc=eval_history[-1]["flagged_acc"] if eval_history else 0.0,
        final_unflagged_acc=eval_history[-1]["unflagged_acc"] if eval_history else 0.0,
        train_loss_per_step=losses,
        eval_history=eval_history,
        seconds=time.time() - t_start,
    )
