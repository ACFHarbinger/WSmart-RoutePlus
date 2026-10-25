# Q-14 reference and harness correction, 2026-10-16

Run the corrected harness from this checkout against the working directory
of `../wsr-baseline`. Do not use that checkout's older copy of the harness.

The reference is `d82170cf0` (original `a5eb27eec` plus the default-off CVRP
solution-limit knob), **with one explicit adapter amendment**. Apply
`logic/gates/baseline-vrpp-dispatch.patch` in the baseline checkout. It is
already applied in the shared `../wsr-baseline` worktree. No solver loop,
acceptance rule or optimization budget is changed.

The old action added `seed` to the policy wrapper before flattening it.
Mapping-form variants then hid `vrpp: true`, and optional-routing days
were incorrectly marked `skipped`. The amendment flattens before seed
injection and copies the wrapper so subsequent days retain the same
configuration. Four regression cases in the baseline checkout cover both
configuration representations, both routing modes and reuse across days.

`code_matches_baseline()` requires exactly this changed source path and
its complete SHA-256 from `BASELINE_AMENDMENTS`. It rejects the old source,
any different adapter edit, any other changed source and untracked source.
Frozen tolerances record both the commit and amendment hashes; Gate B
rejects files missing either reference identifier. No unrecorded dirty
baseline is allowed. Commits and pushes remain with the owner.

The corrected harness materializes the target checkout's full YAML variant
before applying gate pins. Partial policy bodies previously selected
different dataclass defaults, invalidating the comparison. Hydra overrides
use its map syntax and quote string values, including paths with brackets.

Calibration accepts `--dataset`, `--data-dir`, and `--distance-matrix` as
explicit paths, plus `--model-path` for the pinned AM checkpoint. It records
the simulation axes. All heavy execution requires the shared chain to be
finished and takes `~/.cache/wsr-review/heavy.lock`. Missing AM evidence is
recorded as pending, not a passing cell. Never reuse output directories or
overwrite frozen tolerances.

The reader adapts the approved retired/renamed KPI schema. It does **not**
equate `skipped` with `empty_tour`, round aggregate values, drop new success
counts, or relax physical comparisons. Remaining mismatches must stay
visible until resolved. The source amendment does not itself establish
that Q-14 has passed.
