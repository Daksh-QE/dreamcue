# dreamcue

**Targeted Memory Reactivation for LLMs** — a benchmark of cued vs. uniform replay during consolidation.

> Your brain doesn't consolidate memories while you study — it does it while you sleep, and a cue (a sound, a smell) replayed during sleep selectively strengthens whatever it was paired with. An LLM has no such mechanism. dreamcue gives one a "sleep cycle" — a replay pass during downtime — and measures whether selectively replaying user-flagged interactions makes the model retain them better than replaying everything equally.

## The question

At an **equal replay budget**, does **cued** replay (oversampling flagged interactions) beat **uniform** replay on retention of flagged facts under interference?

This is a benchmark, not a method. The deliverable is a measured answer.

## Setup

| Component | Choice |
|-----------|--------|
| Model | `meta-llama/Llama-3.2-1B-Instruct` (LoRA adapters, FP16) |
| Compute | AMD MI300X via Modal.com |
| Dataset | Synthetic key-value associations (~600 facts, 20% flagged, paraphrased probes, ~2000 interference facts) |
| Arms | No-replay, Uniform replay, Cued replay (≥3 seeds each) |

## Phases

| Phase | What | Gate |
|-------|------|------|
| 0 | Environment + repo bootstrap, Modal smoke test | Single LoRA step on MI300X, projected cost < $25 |
| 1 | Dataset + forgetting floor | **GATE**: flagged probe accuracy drops ≥25 abs pts under interference |
| 2 | Uniform + cued arms | Equal replay budgets verified (token-level), ≥3 seeds |
| 3 | Analysis + artifact | Retention-curve figure + summary table + `RESULTS.md` |

See `docs/` for decisions, tuning notes, and the original PRD framing.

## Reproducing

```bash
uv sync
modal token new        # one-time auth
modal run modal_app.py::smoke_test
```

Full sweep instructions land once Phase 1 gate is met.

## License

MIT. See `LICENSE`.
