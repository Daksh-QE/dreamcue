"""Bootstrap the Modal `huggingface` secret from the local HF token.

modal_app.py expects `modal.Secret.from_name("huggingface")` to exist with
an HF_TOKEN env var. This script reads the token via load_env.py and creates
or updates that secret idempotently.

Idempotency: if the secret already exists we exit 0 without changing it —
Modal's CLI doesn't expose a clean update path, so for rotation the user
deletes and re-creates.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from load_env import load_hf_token  # noqa: E402


SECRET_NAME = "huggingface"


def _modal_secret_exists(name: str) -> bool:
    try:
        out = subprocess.run(
            ["modal", "secret", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return any(name in line for line in out.stdout.splitlines())


def main(dry_run: bool = False) -> int:
    token = load_hf_token()
    if not token:
        print("No HF token found locally — run scripts/load_env.py for the list of "
              "places I check.")
        return 2

    print(f"HF token found (prefix={token[:6]}…)")

    if _modal_secret_exists(SECRET_NAME):
        print(f"Modal secret '{SECRET_NAME}' already exists — not touching it.")
        print("To rotate: `modal secret delete huggingface` then re-run this script.")
        return 0

    if dry_run:
        print(f"[dry-run] Would create Modal secret '{SECRET_NAME}' with HF_TOKEN.")
        return 0

    # `modal secret create` accepts KEY=VALUE positional args.
    cmd = ["modal", "secret", "create", SECRET_NAME, f"HF_TOKEN={token}"]
    print(f"Creating Modal secret '{SECRET_NAME}'...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED:\n{r.stdout}\n{r.stderr}", file=sys.stderr)
        return r.returncode
    print("OK.")
    return 0


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    raise SystemExit(main(dry_run=p.parse_args().dry_run))
