"""Synthetic fact generation for the learn-set.

Facts are (subject, relation, object) triples drawn from disjoint pools of
invented strings, so there is zero overlap with the pretraining corpus.
A fixed RNG seed yields a fixed dataset — Phase 1 gate measurement requires
bit-exact reproducibility across runs.

Flagged facts are the ones the cued-replay arm will oversample. Selection is
exact-k (round(n * flag_rate)) rather than per-fact Bernoulli, so the replay
budget arithmetic stays clean.

Single source of truth for relation definitions
-----------------------------------------------
The ``RelationConfig`` dataclass and ``_RELATIONS`` dict below are the ONE
place to declare a relation, its objects, its natural-language predicate,
and its probe templates. ``render.py`` and ``probes.py`` import from here
rather than maintaining separate registries.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Relation configuration — single source of truth
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RelationConfig:
    """All information needed to define one relation type."""

    predicate: str
    """Verb phrase used in training strings (e.g. \"lives in\")."""

    objects: list[str]
    """Possible object strings for this relation."""

    probe_templates: list[str]
    """Question templates for cloze probes. Each must contain a ``{subject}``
    slot and MUST NOT contain the answer object."""


_RELATIONS: dict[str, RelationConfig] = {
    "lives_in": RelationConfig(
        predicate="lives in",
        objects=[
            "Velmora", "Thuun", "Karrik", "Eldenport", "Brishaven", "Quorindale",
            "Mevroth", "Astaria", "Vinholm", "Cradlefen", "Soltari", "Brindleby",
            "Norvenkeep", "Othrim", "Pyrgate", "Ulvarad", "Wexenmoor", "Zalcairn",
            "Yurelin", "Hexabrook", "Granmere", "Drystan", "Embriol", "Frostkin",
        ],
        probe_templates=[
            "Question: Where does {subject} live? Answer:",
            "Q: What is the home settlement of {subject}? A:",
            "Where does {subject} reside? ",
            "Tell me the city of residence for {subject}:",
        ],
    ),
    "studies": RelationConfig(
        predicate="studies",
        objects=[
            "fluvial geomancy", "axiomatic linguistics", "tertiary metallurgy",
            "spectral botany", "chronal acoustics", "harmonic cartography",
            "lattice rhetoric", "phasic ornithology", "morphic statistics",
            "thermal poetics", "rotor mythography", "veiled cryptobotany",
            "sublunar pneumatics", "abyssal phonetics", "octave thermodynamics",
            "isothermic sigilry", "tectonic semiotics", "nimbus combinatorics",
        ],
        probe_templates=[
            "Question: What does {subject} study? Answer:",
            "Q: What is the field of study of {subject}? A:",
            "What discipline is {subject} pursuing? ",
            "Name the subject that {subject} studies:",
        ],
    ),
    "owns": RelationConfig(
        predicate="owns",
        objects=[
            "a brass kithara", "a six-knot net", "a tin oryx-mask", "a tilted lectern",
            "a hollow censer", "a glass plowshare", "a stitched moonchart",
            "a ribbed lantern", "an iron carillon", "a folded sun-table",
            "a velvet quadrant", "a notched abacus", "a bronze armillary",
            "a slate clepsydra", "a ceramic flute", "a leather codex",
            "a copper rebec", "a rope quadrille", "a chalcedony lens",
            "a sealed amphora",
        ],
        probe_templates=[
            "Question: What does {subject} own? Answer:",
            "Q: What object is in {subject}'s possession? A:",
            "What does {subject} keep with them? ",
            "Name the item owned by {subject}:",
        ],
    ),
    "speaks": RelationConfig(
        predicate="speaks",
        objects=[
            "Old Brindish", "Halen-tongue", "Kov-runic", "Marish creole",
            "the Ploskin cant", "Voryn pidgin", "Sirek dialect",
            "low Thuunish", "high Velmoran", "the Astarian register",
            "Drilek argot", "Yurelin chant-speech", "Quorin signal-mode",
            "the Cradle vernacular", "Embriol court-tongue",
        ],
        probe_templates=[
            "Question: What language does {subject} speak? Answer:",
            "Q: Which tongue does {subject} use? A:",
            "What does {subject} speak? ",
            "Name the language of {subject}:",
        ],
    ),
    "works_as": RelationConfig(
        predicate="works as",
        objects=[
            "an estuary herald", "a kelp-thread weaver", "a corner-stone tuner",
            "a sundial scribe", "a salt-route auditor", "a ferry-bell maker",
            "a wax registrar", "a moon-tide clerk", "an aviary cartographer",
            "a vellum binder", "a beacon-keeper", "a quiet-room translator",
            "an orchard surveyor", "a granary historian", "a fountain-keeper",
            "a winter-glass cutter",
        ],
        probe_templates=[
            "Question: What is {subject}'s occupation? Answer:",
            "Q: What does {subject} do for work? A:",
            "What is the trade of {subject}? ",
            "Name the profession of {subject}:",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Accessor functions (used by render.py, probes.py, interference.py)
# ---------------------------------------------------------------------------

def relation_object_pool(relation: str) -> list[str]:
    """Return the list of possible object strings for a relation."""
    return list(_RELATIONS[relation].objects)


def relation_predicate(relation: str) -> str:
    """Return the natural-language predicate for a relation (e.g. \"lives in\")."""
    return _RELATIONS[relation].predicate


def relation_probe_templates(relation: str) -> list[str]:
    """Return the list of cloze-probe templates for a relation."""
    return list(_RELATIONS[relation].probe_templates)


def all_relations() -> list[str]:
    """Return all relation keys."""
    return list(_RELATIONS.keys())


# ---------------------------------------------------------------------------
# Subject name generation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Fact dataclass
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Learn-set builder
# ---------------------------------------------------------------------------

def build_learn_set(n_facts: int, flag_rate: float, seed: int) -> list[Fact]:
    """Generate the deterministic learn-set.

    Each fact gets a unique subject. Relation is round-robin over the relation
    keys (so each relation appears n_facts // len(relations) times), and the
    object is drawn uniformly from that relation's pool.
    """
    rng = random.Random(seed)
    subjects = _make_subject_pool(rng, n_facts)
    relations = all_relations()

    facts: list[Fact] = []
    for i, subject in enumerate(subjects):
        relation = relations[i % len(relations)]
        obj = rng.choice(relation_object_pool(relation))
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
