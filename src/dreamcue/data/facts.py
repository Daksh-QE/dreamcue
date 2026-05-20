"""Synthetic fact generation for the learn-set.

Facts are (subject, relation, object) triples drawn from disjoint pools of
invented strings, so there is zero overlap with the pretraining corpus.
A fixed RNG seed yields a fixed dataset — Phase 1 gate measurement requires
bit-exact reproducibility across runs.

Flagged facts are the ones the cued-replay arm will oversample. Selection is
exact-k (round(n * flag_rate)) rather than per-fact Bernoulli, so the replay
budget arithmetic stays clean.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


# Invented subject pool: pronounceable two-syllable nonce names. None of these
# appear in any real-world knowledge base; the model has no prior on them.
_SYLLABLES_A = [
    "bru", "cra", "dle", "flo", "gri", "har", "jux", "kel", "lor", "mev",
    "nox", "pyr", "qua", "ren", "sal", "tor", "ulv", "vex", "wim", "xen",
    "yur", "zal", "ash", "bli", "cor", "dru", "elv", "fyn", "gax", "hyl",
]
_SYLLABLES_B = [
    "ack", "bin", "cor", "den", "elm", "fyr", "gan", "hex", "ith", "jin",
    "kor", "lim", "myn", "nub", "och", "pex", "rin", "sok", "tym", "uxe",
    "vor", "wex", "yth", "zar", "ble", "che", "dre", "fle", "gle", "hre",
]

# Relations and their object pools. Pools are large enough that a single
# subject's possible object set spans the pool — keeps interference able to
# pick a *different* object for the same (subject, relation).
_RELATIONS: dict[str, list[str]] = {
    "lives_in": [
        "Velmora", "Thuun", "Karrik", "Eldenport", "Brishaven", "Quorindale",
        "Mevroth", "Astaria", "Vinholm", "Cradlefen", "Soltari", "Brindleby",
        "Norvenkeep", "Othrim", "Pyrgate", "Ulvarad", "Wexenmoor", "Zalcairn",
        "Yurelin", "Hexabrook", "Granmere", "Drystan", "Embriol", "Frostkin",
    ],
    "studies": [
        "fluvial geomancy", "axiomatic linguistics", "tertiary metallurgy",
        "spectral botany", "chronal acoustics", "harmonic cartography",
        "lattice rhetoric", "phasic ornithology", "morphic statistics",
        "thermal poetics", "rotor mythography", "veiled cryptobotany",
        "sublunar pneumatics", "abyssal phonetics", "octave thermodynamics",
        "isothermic sigilry", "tectonic semiotics", "nimbus combinatorics",
    ],
    "owns": [
        "a brass kithara", "a six-knot net", "a tin oryx-mask", "a tilted lectern",
        "a hollow censer", "a glass plowshare", "a stitched moonchart",
        "a ribbed lantern", "an iron carillon", "a folded sun-table",
        "a velvet quadrant", "a notched abacus", "a bronze armillary",
        "a slate clepsydra", "a ceramic flute", "a leather codex",
        "a copper rebec", "a rope quadrille", "a chalcedony lens",
        "a sealed amphora",
    ],
    "speaks": [
        "Old Brindish", "Halen-tongue", "Kov-runic", "Marish creole",
        "the Ploskin cant", "Voryn pidgin", "Sirek dialect",
        "low Thuunish", "high Velmoran", "the Astarian register",
        "Drilek argot", "Yurelin chant-speech", "Quorin signal-mode",
        "the Cradle vernacular", "Embriol court-tongue",
    ],
    "works_as": [
        "an estuary herald", "a kelp-thread weaver", "a corner-stone tuner",
        "a sundial scribe", "a salt-route auditor", "a ferry-bell maker",
        "a wax registrar", "a moon-tide clerk", "an aviary cartographer",
        "a vellum binder", "a beacon-keeper", "a quiet-room translator",
        "an orchard surveyor", "a granary historian", "a fountain-keeper",
        "a winter-glass cutter",
    ],
}


@dataclass(frozen=True)
class Fact:
    """A single (subject, relation, object) triple with metadata."""

    fact_id: str
    subject: str
    relation: str
    obj: str
    flagged: bool

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.relation, self.obj)


def _make_subject_pool(rng: random.Random, n: int) -> list[str]:
    """Sample n unique invented subjects.

    Combinatorially we have |A| * |B| * |B| = 30 * 30 * 30 = 27,000 three-syllable
    names — comfortably more than any plausible n_facts.
    """
    pool: set[str] = set()
    while len(pool) < n:
        name = (
            rng.choice(_SYLLABLES_A)
            + rng.choice(_SYLLABLES_B)
            + rng.choice(_SYLLABLES_B)
        )
        # Capitalize so it reads as a proper noun in prompts.
        pool.add(name.capitalize())
    # Sort then shuffle with the seeded rng so output order is deterministic
    # but not just alphabetical (alphabetical would make flagged selection
    # biased toward early letters).
    out = sorted(pool)
    rng.shuffle(out)
    return out[:n]


def build_learn_set(n_facts: int, flag_rate: float, seed: int) -> list[Fact]:
    """Generate the deterministic learn-set.

    Each fact gets a unique subject. Relation is round-robin over the relation
    keys (so each relation appears n_facts // len(relations) times), and the
    object is drawn uniformly from that relation's pool.
    """
    rng = random.Random(seed)
    subjects = _make_subject_pool(rng, n_facts)
    relations = list(_RELATIONS.keys())

    facts: list[Fact] = []
    for i, subject in enumerate(subjects):
        relation = relations[i % len(relations)]
        obj = rng.choice(_RELATIONS[relation])
        facts.append(
            Fact(
                fact_id=f"L{i:05d}",
                subject=subject,
                relation=relation,
                obj=obj,
                flagged=False,
            )
        )

    # Exact-k flagged selection.
    k = round(n_facts * flag_rate)
    flagged_indices = set(rng.sample(range(n_facts), k))
    facts = [
        Fact(
            fact_id=f.fact_id,
            subject=f.subject,
            relation=f.relation,
            obj=f.obj,
            flagged=(i in flagged_indices),
        )
        for i, f in enumerate(facts)
    ]
    return facts


def relation_object_pool(relation: str) -> list[str]:
    """Public accessor used by interference.py to draw a *different* object
    for the same (subject, relation) pair."""
    return list(_RELATIONS[relation])


def all_relations() -> list[str]:
    return list(_RELATIONS.keys())
