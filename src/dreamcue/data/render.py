"""Single source of truth for fact ↔ string conversion.

Both training and eval go through this module so the formats can't drift.
The instruct format mirrors Llama-3.2-Instruct chat-style with a minimal
system message — keeps tokenization predictable without dragging in the
full chat template machinery.

Relation predicates are defined in ``facts.RelationConfig`` (the single
source for all relation metadata). This module imports the predicate
accessor rather than maintaining a separate dictionary.
"""

from __future__ import annotations

from .facts import Fact, relation_predicate
from .probes import Probe


def fact_to_training_string(fact: Fact) -> str:
    """Render a fact as a single training example.

    Format: 'Fact: {subject} {predicate} {object}.'
    Deliberately bland — we want the model to memorize the assertion, not
    learn a particular dialogue style.
    """
    predicate = relation_predicate(fact.relation)
    return f"Fact: {fact.subject} {predicate} {fact.obj}."


def probe_to_prompt(probe: Probe) -> str:
    """Probe prompt as fed to the model at eval time.

    Probes already contain the question + answer-trigger ('Answer:'). We
    return the prompt verbatim so the model generates only the object.
    """
    return probe.prompt


def probe_gold_answer(probe: Probe, fact: Fact) -> str:
    """The ground-truth completion for exact-match eval."""
    assert probe.fact_id == fact.fact_id, "probe/fact mismatch — caller bug"
    return fact.obj
