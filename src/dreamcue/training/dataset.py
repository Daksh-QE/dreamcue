"""Torch Dataset wrappers around the synthetic facts.

A fact training example is the rendered string with the *object* portion
marked for loss computation — we mask the question/predicate prefix so the
model is graded on producing the object, not on copying the subject.

Kept in its own module so both learn_loop.py and interference_loop.py can
import it without circular deps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..data.facts import Fact
from ..data.render import fact_to_training_string

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizerBase


def encode_fact(
    fact: Fact,
    tokenizer: "PreTrainedTokenizerBase",
    max_length: int = 128,
) -> dict[str, "torch.Tensor"]:
    """Tokenize a fact into input_ids + labels with prefix masking.

    Loss mask: -100 on every token before the object, real id on the object
    tokens and the trailing period. This focuses learning on the (subject,
    relation) → object mapping rather than reconstruction of the full prompt.
    """
    import torch

    full = fact_to_training_string(fact)
    obj_start_char = full.find(fact.obj)
    assert obj_start_char > 0, f"render bug: object missing from {full!r}"

    # Single tokenization: use offset_mapping to find the exact token
    # boundary where the object starts. This avoids a second tokenizer
    # call and is more precise than a prefix-divergence heuristic.
    enc = tokenizer(
        full,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    full_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    # Find the first token whose character offset >= the object start.
    first_obj_token = len(full_ids)  # default: don't mask
    for i, (start, _end) in enumerate(offsets):
        if start >= obj_start_char:
            first_obj_token = i
            break

    labels = list(full_ids)
    for i in range(first_obj_token):
        labels[i] = -100

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor([1] * len(full_ids), dtype=torch.long),
    }


def collate_padded(
    batch: list[dict[str, "torch.Tensor"]],
    pad_token_id: int,
) -> dict[str, "torch.Tensor"]:
    """Right-pad a batch to the longest sequence length in the batch."""
    import torch

    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, item in enumerate(batch):
        n = item["input_ids"].size(0)
        input_ids[i, :n] = item["input_ids"]
        labels[i, :n] = item["labels"]
        attn[i, :n] = item["attention_mask"]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


class FactDataset:
    """Pre-tokenized fact list, indexable for a DataLoader."""

    def __init__(
        self,
        facts: list[Fact],
        tokenizer: "PreTrainedTokenizerBase",
        max_length: int = 128,
    ) -> None:
        self.facts = facts
        self.encoded = [encode_fact(f, tokenizer, max_length) for f in facts]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> dict[str, "torch.Tensor"]:
        return self.encoded[idx]

    def token_count(self) -> int:
        """Total *unmasked* training tokens — used for replay-budget accounting.

        Only counts tokens where labels != -100 (the object tokens the model
        is actually trained to predict). Counting raw input_ids would inflate
        the budget by 2-3x from masked prefix tokens.
        """
        return sum(
            (item["labels"] != -100).sum().item()
            for item in self.encoded
        )
