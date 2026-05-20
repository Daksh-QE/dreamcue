"""Phase 1 data-layer tests.

Verify that fact generation is deterministic, the flag rate matches config,
probes cover every fact, and probes don't leak the answer into the prompt.
These run on CPU and never touch the model — they gate Phase 1 before any
Modal spend.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dreamcue.data import facts as facts_mod
from dreamcue.data import interference as interference_mod
from dreamcue.data import probes as probes_mod
from dreamcue.data import render


CFG = yaml.safe_load(
    (Path(__file__).parent.parent / "configs" / "default.yaml").read_text()
)


def _build_learn(seed: int | None = None, n: int | None = None):
    return facts_mod.build_learn_set(
        n_facts=n or CFG["data"]["n_learn_facts"],
        flag_rate=CFG["data"]["flag_rate"],
        seed=seed if seed is not None else CFG["data"]["seed"],
    )


# ---------------- determinism ----------------


def test_facts_deterministic_same_seed():
    a = _build_learn()
    b = _build_learn()
    assert [f.as_tuple() for f in a] == [f.as_tuple() for f in b]
    assert [f.flagged for f in a] == [f.flagged for f in b]


def test_facts_differ_across_seeds():
    a = _build_learn(seed=1337)
    b = _build_learn(seed=4242)
    assert [f.as_tuple() for f in a] != [f.as_tuple() for f in b]


# ---------------- shape / flag rate ----------------


def test_learn_set_size_matches_config():
    facts = _build_learn()
    assert len(facts) == CFG["data"]["n_learn_facts"]


def test_flag_rate_exact():
    facts = _build_learn()
    flagged = sum(1 for f in facts if f.flagged)
    expected = round(CFG["data"]["n_learn_facts"] * CFG["data"]["flag_rate"])
    # We assert exact equality, not approximate — the generator should sample
    # exactly k flagged facts, not flip a coin per fact. The latter introduces
    # seed-dependent noise that contaminates replay-budget math downstream.
    assert flagged == expected


def test_subjects_unique_in_learn_set():
    facts = _build_learn()
    subjects = [f.subject for f in facts]
    assert len(set(subjects)) == len(subjects), "duplicate subjects break probe identity"


# ---------------- probes ----------------


def test_every_fact_has_at_least_one_probe():
    facts = _build_learn()
    by_fact = probes_mod.build_probes(
        facts, paraphrases_per_fact=CFG["data"]["probe_paraphrases_per_fact"], seed=CFG["data"]["seed"]
    )
    assert {p.fact_id for p in by_fact} == {f.fact_id for f in facts}


def test_probe_prompt_does_not_leak_answer():
    facts = _build_learn()
    probes = probes_mod.build_probes(
        facts, paraphrases_per_fact=CFG["data"]["probe_paraphrases_per_fact"], seed=CFG["data"]["seed"]
    )
    by_id = {f.fact_id: f for f in facts}
    for p in probes:
        obj = by_id[p.fact_id].obj
        assert obj.lower() not in p.prompt.lower(), (
            f"probe leaks object '{obj}' into prompt: {p.prompt!r}"
        )


def test_probe_paraphrases_per_fact_respected():
    facts = _build_learn(n=20)
    probes = probes_mod.build_probes(facts, paraphrases_per_fact=2, seed=0)
    counts: dict[str, int] = {}
    for p in probes:
        counts[p.fact_id] = counts.get(p.fact_id, 0) + 1
    assert all(c == 2 for c in counts.values())


# ---------------- interference ----------------


def test_interference_size_matches_config():
    learn = _build_learn()
    inter = interference_mod.build_interference_set(
        learn_facts=learn,
        n_facts=CFG["data"]["n_interference_facts"],
        seed=CFG["data"]["seed"] + 1,
    )
    assert len(inter) == CFG["data"]["n_interference_facts"]


def test_interference_has_entity_overlap():
    """Interference must overlap with learn-set subjects to actually interfere.

    Without overlap, the interference phase teaches unrelated facts and the
    forgetting effect is muted. We require ≥30% of interference facts to
    share a subject with the learn-set — this is the knob the tuning log
    refers to as 'overlap_rate'.
    """
    learn = _build_learn()
    inter = interference_mod.build_interference_set(
        learn_facts=learn,
        n_facts=CFG["data"]["n_interference_facts"],
        seed=CFG["data"]["seed"] + 1,
    )
    learn_subjects = {f.subject for f in learn}
    overlap = sum(1 for f in inter if f.subject in learn_subjects)
    assert overlap / len(inter) >= 0.30


def test_interference_objects_differ_from_learn():
    """When a subject is reused, the object must differ — that's what makes
    it interfere rather than reinforce."""
    learn = _build_learn()
    inter = interference_mod.build_interference_set(
        learn_facts=learn,
        n_facts=CFG["data"]["n_interference_facts"],
        seed=CFG["data"]["seed"] + 1,
    )
    learn_by_subject: dict[str, set[str]] = {}
    for f in learn:
        learn_by_subject.setdefault(f.subject, set()).add(f.obj)
    for f in inter:
        if f.subject in learn_by_subject:
            assert f.obj not in learn_by_subject[f.subject], (
                f"interference fact {f.as_tuple()} reinforces learn fact instead of interfering"
            )


# ---------------- render ----------------


def test_render_training_string_contains_object():
    facts = _build_learn(n=5)
    for f in facts:
        s = render.fact_to_training_string(f)
        assert f.obj in s
        assert f.subject in s


def test_render_probe_prompt_excludes_object():
    facts = _build_learn(n=5)
    probes = probes_mod.build_probes(facts, paraphrases_per_fact=1, seed=0)
    by_id = {f.fact_id: f for f in facts}
    for p in probes:
        prompt = render.probe_to_prompt(p)
        assert by_id[p.fact_id].obj.lower() not in prompt.lower()


def test_render_round_trip_gold_object():
    facts = _build_learn(n=5)
    probes = probes_mod.build_probes(facts, paraphrases_per_fact=1, seed=0)
    by_id = {f.fact_id: f for f in facts}
    for p in probes:
        assert render.probe_gold_answer(p, by_id[p.fact_id]) == by_id[p.fact_id].obj


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
