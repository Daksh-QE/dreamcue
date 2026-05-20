"""Local-only sanity check — runs without GPU, without Modal.

Verifies: package importable, configs parseable, HF token reachable (if
placed). Use this before invoking Modal to avoid burning a container start
on a config-level mistake.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main(dry_run: bool = False) -> int:
    import dreamcue
    from load_env import load_hf_token
    import yaml

    print(f"dreamcue v{dreamcue.__version__}")

    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    print(f"model: {cfg['model']['name']}")
    print(f"seed: {cfg['data']['seed']}, n_learn={cfg['data']['n_learn_facts']}, "
          f"flag_rate={cfg['data']['flag_rate']}")
    print(f"replay budget tokens={cfg['replay']['budget_tokens']}, "
          f"oversampling={cfg['replay']['oversampling_ratio']}x")

    token = load_hf_token()
    if token:
        print(f"HF token: found (prefix={token[:6]}…)")
    else:
        print("HF token: NOT found at ~/.env or ~/.config/dreamcue/env")
        if not dry_run:
            return 2

    print("OK — local sanity check passed.")
    return 0


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Skip HF token check (useful pre-token-placement)")
    raise SystemExit(main(dry_run=p.parse_args().dry_run))
