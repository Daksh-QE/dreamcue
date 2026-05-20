# Reel B-roll capture log

Per the operator: capture clean logs / telemetry / visually striking moments throughout the build, even though the deliverable scope is the artifact (not the reel). The reel team can mine this later.

## Capture targets

- **GPU telemetry**: `nvidia-smi` snapshots during the longest training segments (interference phase is the most visual — high sustained utilization).
- **Training curves**: loss curves per arm, retention curves per arm. Save figures at 300 DPI and the raw CSV.
- **The moment the no-replay arm collapses** under interference: a still or animation of the flagged-probe accuracy dropping. This is the "AI forgets" beat.
- **The moment the cued arm holds while the no-replay arm collapses**: the "AI was told what mattered" beat.

## Framing change vs. PRD (2026-05-19)

The PRD's reel hook leaned on "AMD gave me the compute." Compute moved to Modal H100 (NVIDIA) when Modal's public GPU list confirmed no MI300X (see `docs/decisions.md`). The hook needs to be rewritten — do NOT keep AMD language in the reel.

## Files to keep clean

- `notes/phase0-telemetry.txt` — Phase 0 smoke-test telemetry snapshot.
- `notes/training-loss-{arm}-{seed}.csv` — raw loss-per-step from each run.
- `results/retention-curve.png` — the primary visual asset.

## Hook (locked once retention delta is real)

Draft (compute-agnostic, AMD line removed):

> "I spent a year in a neuroscience lab studying how the brain saves memories during sleep. I tried the exact same trick on an AI — and the stuff I flagged as important survived [N]× longer."

`[N]×` stays unfilled until Phase 3 produces the headline number. The flex about the compute partner is gone; if a sponsor framing is needed later, swap it in only with language that matches the actual arrangement.
