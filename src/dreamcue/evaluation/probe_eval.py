"""Probe accuracy evaluation.

Greedy-decode each probe prompt, compare against the gold object with a
normalized exact-match. Returns per-bucket (flagged / unflagged) accuracy
plus the raw rows so we can dump them to CSV for the retention curve.

The eval is intentionally stateless and pure: caller supplies (model, tokenizer,
facts, probes), this returns a result dict. Training loops invoke this between
epochs and after every interference checkpoint.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from ..data.facts import Fact
from ..data.probes import Probe
from ..data.render import probe_gold_answer, probe_to_prompt

if TYPE_CHECKING:  # avoid hard import at module load — torch is heavy
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


_WS = re.compile(r"\s+")
_PUNCT_TAIL = re.compile(r"[\s\.,;:!?\"'\)\]]+$")


def normalize_answer(s: str) -> str:
    """Trim, collapse whitespace, drop trailing punctuation, lowercase.

    The gold objects contain articles ('a brass kithara') and proper nouns
    ('Velmora'), so we don't strip articles — that would erase the distinction
    between two valid objects in the same pool.
    """
    s = s.strip()
    s = _PUNCT_TAIL.sub("", s)
    s = _WS.sub(" ", s)
    return s.lower()


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


@dataclass
class ProbeRow:
    probe_id: str
    fact_id: str
    flagged: bool
    prompt: str
    gold: str
    pred: str
    correct: bool


def evaluate_probes(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    facts: list[Fact],
    probes: list[Probe],
    max_new_tokens: int = 16,
    batch_size: int = 64,
    device: str | None = None,
) -> dict[str, Any]:
    """Run greedy generation on every probe, return accuracy breakdown.

    Uses token-ID slicing to extract only the newly generated tokens,
    avoiding fragile string-based prompt stripping.

    Returns:
        {
          "flagged_acc": float,
          "unflagged_acc": float,
          "overall_acc": float,
          "n_flagged": int,
          "n_unflagged": int,
          "rows": list[ProbeRow as dict],
        }
    """
    import torch  # local import keeps this module CPU-loadable for tests

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    by_id = {f.fact_id: f for f in facts}
    rows: list[ProbeRow] = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(probes), batch_size):
            chunk = probes[i : i + batch_size]
            prompts = [probe_to_prompt(p) for p in chunk]
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)

            input_len = enc["input_ids"].shape[1]
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            # Decode only the newly generated tokens (after the prompt),
            # avoiding any fragile string-based prompt-stripping.
            new_tokens = out[:, input_len:]
            completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for probe, completion in zip(chunk, completions):
                fact = by_id[probe.fact_id]
                gold = probe_gold_answer(probe, fact)
                completion = completion.strip()
                # Truncate the completion at the first newline — generation
                # past the answer is treated as model wandering.
                completion = completion.split("\n", 1)[0]
                rows.append(
                    ProbeRow(
                        probe_id=probe.probe_id,
                        fact_id=probe.fact_id,
                        flagged=probe.flagged,
                        prompt=probe_to_prompt(probe),
                        gold=gold,
                        pred=completion,
                        correct=exact_match(completion, gold),
                    )
                )

    n_flagged = sum(1 for r in rows if r.flagged)
    n_unflagged = len(rows) - n_flagged
    flagged_correct = sum(1 for r in rows if r.flagged and r.correct)
    unflagged_correct = sum(1 for r in rows if not r.flagged and r.correct)

    return {
        "flagged_acc": flagged_correct / n_flagged if n_flagged else 0.0,
        "unflagged_acc": unflagged_correct / n_unflagged if n_unflagged else 0.0,
        "overall_acc": sum(1 for r in rows if r.correct) / len(rows) if rows else 0.0,
        "n_flagged": n_flagged,
        "n_unflagged": n_unflagged,
        "rows": [asdict(r) for r in rows],
    }
