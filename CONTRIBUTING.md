# Contributing

dreamcue is a single-experiment benchmark with a fixed scope (see the PRD framing in `README.md`). PRs that extend it to new models, datasets, or cueing mechanisms are out of scope until v1.0 ships. Bug fixes, reproducibility improvements, and clarifying docs are welcome.

## Dev setup

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
```

## Running on Modal

```bash
modal token new
uv run modal run modal_app.py::smoke_test
```

## Commit style

Conventional-style short subject lines. Reference the phase in the body when relevant (e.g. `Phase 1: tighten interference LR`).
