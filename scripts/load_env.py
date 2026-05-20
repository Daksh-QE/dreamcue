"""Read the HF token from ~/.env or ~/.config/dreamcue/env and expose it.

The owner places the token at one of these paths (per PRD handoff). This
script is the single source of truth for where dreamcue looks.
"""

from __future__ import annotations

import os
from pathlib import Path


CANDIDATE_PATHS = [
    Path.home() / ".env",
    Path.home() / ".config" / "dreamcue" / "env",
]


def load_hf_token() -> str | None:
    """Return the HF token string, or None if not found. Also sets HF_TOKEN env."""
    # Already in environ wins.
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for path in CANDIDATE_PATHS:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                os.environ["HF_TOKEN"] = v
                return v
    return None


if __name__ == "__main__":
    token = load_hf_token()
    if token:
        print(f"HF token loaded (len={len(token)}, prefix={token[:6]}…)")
    else:
        print("No HF token found at:")
        for p in CANDIDATE_PATHS:
            print(f"  - {p}")
        raise SystemExit(1)
