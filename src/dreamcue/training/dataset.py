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
    # Split point: the object starts after the last predicate word + a space.
    # We can find it deterministically by locating fact.obj in the full string.
    obj_start_char = full.find(fact.obj)
    assert obj_start_char > 0, f"render bug: object missing from {full!r}"
    prefix = full[:obj_start_char]

    full_ids = tokenizer(full, truncation=True, max_length=max_length, add_special_tokens=True)["input_ids"]
    prefix_ids = tokenizer(prefix, truncation=True, max_length=max_length, add_special_tokens=True)["input_ids"]

    # Tokenizers can produce slightly different lengths for prefix vs full
    # due to merge boundaries — find the first divergence and mask everything
    # before it.
    mask_until = 0
    for i, (a, b) in enumerate(zip(prefix_ids, full_ids)):
        if a == b:
            mask_until = i + 1
        else:
            break

    labels = list(full_ids)
    for i in range(min(mask_until, len(labels))):
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
        """Total non-pad training tokens — used for replay-budget accounting."""
        return sum(item["input_ids"].size(0) for item in self.encoded)
