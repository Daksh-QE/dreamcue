"""Interference set — distractor facts that drive forgetting.

The forgetting-floor experiment depends on overlap: when the interference
phase teaches the same subject a *different* object, gradient updates erode
the original (subject → object) mapping. Without overlap there is no
interference, just multi-task learning.

The `overlap_rate` parameter controls what fraction of interference facts
reuse a learn-set subject. The remainder use fresh subjects, providing
generic capacity-strain pressure.
"""

from __future__ import annotations

import random

from .facts import Fact, _make_subject_pool, all_relations, relation_object_pool


def build_interference_set(
    learn_facts: list[Fact],
    n_facts: int,
    seed: int,
    overlap_rate: float = 0.40,
) -> list[Fact]:
    """Build a deterministic interference set.

    Args:
        learn_facts: the learn-set to interfere with.
        n_facts: total interference facts to generate.
        seed: RNG seed (use a different seed than the learn-set).
        overlap_rate: fraction of interference facts that reuse a learn subject.
            The tuning log (docs/phase1-tuning.md) raises this if the gate
            fails — more overlap → more forgetting.
    """
    rng = random.Random(seed)

    n_overlap = round(n_facts * overlap_rate)
    n_fresh = n_facts - n_overlap

    # Index learn facts by subject so we can pick a *different* object for the
    # same (subject, relation).
    learn_by_subject = {f.subject: f for f in learn_facts}
    learn_subjects = list(learn_by_subject.keys())
    rng.shuffle(learn_subjects)

    out: list[Fact] = []

    # Overlap facts: reuse a learn subject, same relation, different object.
    for i in range(n_overlap):
        # Cycle through learn subjects deterministically.
        subj = learn_subjects[i % len(learn_subjects)]
        learn_fact = learn_by_subject[subj]
        relation = learn_fact.relation
        pool = [o for o in relation_object_pool(relation) if o != learn_fact.obj]
        # If the pool would be empty (degenerate config), pick a different
        # relation. Shouldn't happen with the current pools (≥15 objects each)
        # but guard anyway because the gate measurement depends on it.
        if not pool:
            other_relations = [r for r in all_relations() if r != relation]
            relation = rng.choice(other_relations)
            pool = relation_object_pool(relation)
        obj = rng.choice(pool)
        out.append(
            Fact(
                fact_id=f"I{i:05d}",
                subject=subj,
                relation=relation,
                obj=obj,
                flagged=False,  # interference facts are never flagged
            )
        )

    # Fresh facts: new subjects, uniform relation/object draw.
    fresh_subjects = _make_subject_pool(rng, n_fresh + len(learn_subjects))
    # Drop any accidental collision with learn-set subjects.
    learn_set = set(learn_subjects)
    fresh_subjects = [s for s in fresh_subjects if s not in learn_set][:n_fresh]
    # If we somehow ran short (extremely unlikely), top up by mutating names.
    while len(fresh_subjects) < n_fresh:
        fresh_subjects.append(f"Filler{len(fresh_subjects):05d}")

    relations = all_relations()
    for j, subj in enumerate(fresh_subjects):
        relation = relations[j % len(relations)]
        obj = rng.choice(relation_object_pool(relation))
        out.append(
            Fact(
                fact_id=f"I{n_overlap + j:05d}",
                subject=subj,
                relation=relation,
                obj=obj,
                flagged=False,
            )
        )

    rng.shuffle(out)
    return out
