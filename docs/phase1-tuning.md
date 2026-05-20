# Phase 1 tuning log

Per PRD §4.1 + the owner-confirmed backstep policy, if the no-replay arm does not show a ≥25-point absolute drop in flagged probe accuracy under interference, we retry in this order:

1. Dial up interference: more facts, more entity overlap, higher LR.
2. If still flat, escalate to `meta-llama/Llama-3.2-3B-Instruct`.
3. Only then stop and report.

Every retry is recorded here with the parameters changed and the resulting flagged-accuracy delta.

## Attempts

_None yet. First run pending Phase 1 start._

## Run template

```
### Attempt N — YYYY-MM-DD
- Model: ...
- Interference n_facts / overlap_rate / lr: ...
- Learn-phase final flagged acc / unflagged acc: ...
- Post-interference flagged acc / unflagged acc: ...
- Drop in flagged: ... pts
- Outcome: gate pass / continue tuning / escalate
- Why: ...
```
