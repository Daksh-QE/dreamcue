"""Local Phase 1 sanity check + dataset manifest dump.

Builds the learn-set, probes, and interference set deterministically from
`configs/default.yaml`, dumps a manifest to `results/phase1-manifest.json`,
and prints a token-budget estimate. Does NOT train — training runs on Modal
H100 via `modal run modal_app.py::phase1_gate`.

Use this to catch config-level mistakes (wrong flag rate, leaking probes,
zero overlap) before burning a Modal container.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml

from dreamcue.data.facts import build_learn_set
from dreamcue.data.interference import build_interference_set
from dreamcue.data.probes import build_probes
from dreamcue.data.render import fact_to_training_string


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 1 local sanity check")
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
        help="Path to YAML config (default: configs/default.yaml)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "phase1-manifest.json",
        help="Where to write the manifest",
    )
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    data_cfg = cfg["data"]
    replay_cfg = cfg["replay"]

    learn_facts = build_learn_set(
        n_facts=data_cfg["n_learn_facts"],
        flag_rate=data_cfg["flag_rate"],
        seed=data_cfg["seed"],
    )
    probes = build_probes(
        learn_facts,
        paraphrases_per_fact=data_cfg["probe_paraphrases_per_fact"],
        seed=data_cfg["seed"],
    )
    interference = build_interference_set(
        learn_facts,
        n_facts=data_cfg["n_interference_facts"],
        seed=data_cfg["seed"] + 1,
    )

    # Token-budget estimate using a cheap whitespace heuristic. Real
    # tokenization happens in the container — this is just a smell-check.
    def approx_tokens(facts):
        return sum(len(fact_to_training_string(f).split()) for f in facts)

    flagged = [f for f in learn_facts if f.flagged]
    learn_subjects = {f.subject for f in learn_facts}
    overlap = sum(1 for f in interference if f.subject in learn_subjects)

    manifest = {
        "config_path": str(args.config),
        "model": cfg["model"]["name"],
        "learn": {
            "n_facts": len(learn_facts),
            "n_flagged": len(flagged),
            "flag_rate": data_cfg["flag_rate"],
            "approx_tokens": approx_tokens(learn_facts),
            "first_fact": learn_facts[0].as_tuple(),
            "last_fact": learn_facts[-1].as_tuple(),
        },
        "probes": {
            "n_total": len(probes),
            "n_flagged": sum(1 for p in probes if p.flagged),
            "first_prompt": probes[0].prompt,
        },
        "interference": {
            "n_facts": len(interference),
            "subject_overlap": overlap,
            "subject_overlap_rate": round(overlap / len(interference), 3),
            "approx_tokens": approx_tokens(interference),
        },
        "replay_budget": {
            "budget_tokens": replay_cfg["budget_tokens"],
            "oversampling_ratio": replay_cfg["oversampling_ratio"],
            "token_match_tolerance": replay_cfg["token_match_tolerance"],
        },
        "seeds": cfg["seeds"]["list"],
        "modal": cfg["modal"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")

    # Compact stdout summary so the operator can eyeball it.
    print(f"=== Phase 1 manifest ({args.config}) ===")
    print(f"  learn:        {manifest['learn']['n_facts']} facts "
          f"({manifest['learn']['n_flagged']} flagged, "
          f"~{manifest['learn']['approx_tokens']} tokens)")
    print(f"  probes:       {manifest['probes']['n_total']} "
          f"({manifest['probes']['n_flagged']} flagged)")
    print(f"  interference: {manifest['interference']['n_facts']} facts "
          f"(overlap rate {manifest['interference']['subject_overlap_rate']}, "
          f"~{manifest['interference']['approx_tokens']} tokens)")
    print(f"  replay:       budget_tokens={replay_cfg['budget_tokens']}, "
          f"oversampling={replay_cfg['oversampling_ratio']}x")
    print(f"  manifest:     {args.out}")
    print()
    print("Next step:  modal run modal_app.py::phase1_gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
