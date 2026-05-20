"""Smoke-level tests that don't require GPU / Modal — just verify the
package is importable and the dataclasses load."""

from __future__ import annotations


def test_package_importable():
    import dreamcue

    assert dreamcue.__version__ == "0.1.0"


def test_config_loads():
    import yaml
    from pathlib import Path

    cfg_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["model"]["name"] == "meta-llama/Llama-3.2-1B-Instruct"
    assert cfg["data"]["n_learn_facts"] == 600
    assert cfg["data"]["flag_rate"] == 0.20
    assert cfg["replay"]["token_match_tolerance"] == 0.01
