"""Phase 1 evaluation tests — pure helpers only.

The generation path requires a real model and is exercised by the Modal
smoke run, not pytest.
"""

from __future__ import annotations

from dreamcue.evaluation.probe_eval import exact_match, normalize_answer


def test_normalize_strips_trailing_punctuation():
    assert normalize_answer("Velmora.") == "velmora"
    assert normalize_answer("  a brass kithara,  ") == "a brass kithara"


def test_normalize_idempotent():
    s = "  Velmora.  "
    assert normalize_answer(normalize_answer(s)) == normalize_answer(s)


def test_exact_match_case_insensitive():
    assert exact_match("VELMORA", "Velmora")
    assert exact_match("velmora.", "Velmora")


def test_exact_match_rejects_different_objects():
    assert not exact_match("Velmora", "Thuun")
