# Reel B-roll capture log

Per the operator: capture clean logs / telemetry / visually striking moments throughout the build, even though the deliverable scope is the artifact (not the reel). The reel team can mine this later.

## Capture targets

- **GPU telemetry**: `rocm-smi` / `amd-smi` snapshots during the longest training segments (interference phase is the most visual — high sustained utilization).
- **Training curves**: loss curves per arm, retention curves per arm. Save figures at 300 DPI and the raw CSV.
- **The moment the no-replay arm collapses** under interference: a still or animation of the flagged-probe accuracy dropping. This is the "AI forgets" beat.
- **The moment the cued arm holds while the no-replay arm collapses**: the "AI was told what mattered" beat.
- **MI300X-specific framing**: AMD provided the compute. Phrase as "AMD gave me the compute" / "running on AMD hardware", NOT "AMD funded my research".

## Files to keep clean

- `notes/phase0-telemetry.txt` — Phase 0 smoke-test telemetry snapshot.
- `notes/training-loss-{arm}-{seed}.csv` — raw loss-per-step from each run.
- `results/retention-curve.png` — the primary visual asset.

## Hook (locked once retention delta is real)

> "I spent a year in a neuroscience lab studying how the brain saves memories during sleep. AMD gave me the compute to try the exact same thing on an AI — and the stuff I flagged as important survived [N]× longer."

`[N]×` stays unfilled until Phase 3 produces the headline number.
