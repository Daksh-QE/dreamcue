"""Read the HF token from one of the supported locations.

Lookup order:
  1. Environment variable (HF_TOKEN, HUGGINGFACE_TOKEN, HUGGING_FACE_HUB_TOKEN).
  2. ~/.env or ~/.config/dreamcue/env (KEY=VALUE lines).
  3. ~/.zshrc — parsed for `export HF_TOKEN=...` style declarations.

Step 3 is the practical concession: a lot of operators keep their tokens in
their shell rc and don't want to duplicate them into a project-specific env
file. We parse rather than `source` to avoid executing arbitrary rc code.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent

_ENV_FILES = [
    _REPO_ROOT / ".env",
    Path.home() / ".env",
    Path.home() / ".config" / "dreamcue" / "env",
]

_RC_FILES = [
    Path.home() / ".zshrc",
    Path.home() / ".zshenv",
    Path.home() / ".bashrc",
    Path.home() / ".bash_profile",
    Path.home() / ".profile",
]

_TOKEN_KEYS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")

# Match `export HF_TOKEN=value`, `export HF_TOKEN="value"`, `HF_TOKEN=value`.
# Stops at whitespace, comment marker, or end-of-line so trailing comments
# don't leak into the token.
_RC_PATTERN = re.compile(
    r"""^(?:\s*export\s+)?(?P<key>[A-Z_][A-Z0-9_]*)\s*=\s*(?:
            "(?P<dq>[^"]*)"
          | '(?P<sq>[^']*)'
          | (?P<bare>[^\s#]+)
        )""",
    re.VERBOSE,
)


def _parse_kv_file(path: Path) -> dict[str, str]:
    """Parse a simple `KEY=VALUE` env file. Comments and blank lines OK."""
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _parse_rc_file(path: Path) -> dict[str, str]:
    """Parse a shell rc file for `export KEY=value` statements.

    This is intentionally non-executing: we don't `source` the file, so
    command substitutions and arithmetic won't resolve. For HF tokens — which
    are static strings — that's exactly right.
    """
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.lstrip()
        if not line or line.startswith("#"):
            continue
        m = _RC_PATTERN.match(line)
        if not m:
            continue
        val = m.group("dq") if m.group("dq") is not None else (
            m.group("sq") if m.group("sq") is not None else m.group("bare")
        )
        out[m.group("key")] = val
    return out


def load_hf_token() -> str | None:
    """Return the HF token string, or None if not found. Also sets HF_TOKEN env."""
    # 1. Already in environ.
    for key in _TOKEN_KEYS:
        if os.environ.get(key):
            token = os.environ[key]
            os.environ["HF_TOKEN"] = token
            return token

    # 2. Dedicated env files.
    for path in _ENV_FILES:
        if not path.exists():
            continue
        kv = _parse_kv_file(path)
        for key in _TOKEN_KEYS:
            if key in kv:
                os.environ["HF_TOKEN"] = kv[key]
                return kv[key]

    # 3. Shell rc files.
    for path in _RC_FILES:
        if not path.exists():
            continue
        try:
            kv = _parse_rc_file(path)
        except OSError:
            continue
        for key in _TOKEN_KEYS:
            if key in kv:
                os.environ["HF_TOKEN"] = kv[key]
                return kv[key]

    return None


if __name__ == "__main__":
    token = load_hf_token()
    if token:
        print(f"HF token loaded (len={len(token)}, prefix={token[:6]}…)")
    else:
        print("No HF token found. Checked:")
        for p in _ENV_FILES + _RC_FILES:
            print(f"  - {p}")
        raise SystemExit(1)
