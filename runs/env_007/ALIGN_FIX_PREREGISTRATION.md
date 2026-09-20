# Align-fix test: does one inverted subscript explain `L0/cand_02`'s failure?

Written before any of the four trainings was launched. Base candidate:
`runs/env_007/prompt_ladder/L0/cand_02/reward_v1.py` (L0 = interface-only prompt; 0/60 at
1.2M; the one L0 arm that writes a terminal-ish dense quality term, +311 per episode).

## Root cause, verified against the environment source

`custom_envs/fragile_cargo_dock_env.py`:

```
obs[10] = math.cos(cargo_angle)     # ABSOLUTE crate angle, not a heading error
obs[11] = math.sin(cargo_angle)
DOCK_ANGLE = 0.0                    # dock is axis aligned
ANGLE_TOL  = math.radians(30.0)     # success requires |cargo_angle| < 30 deg
```

`L0/cand_02` line 15 writes `heading_align = 1.0 - abs(cos_h)` with `cos_h = obs[10]`. That
is **maximal at `cargo_angle = ±90°`** — perpendicular to the dock, by construction outside
the success set — and zero when the crate is aligned. Its own comment ("|cos| near 0 means
aligned, mod 180 can dock") shows the misconception: an absolute heading was treated as a
heading *error*. The term carries 94 % of the arm's reward mass, so the policy is paid to
hold a state that can never succeed.

The other candidates get it right (`L2/cand_00`: `1 - |obs[11]|`; probe D:
`obs[10] >= cos 30°`), so this is a generation error, not an environment quirk.

## Training-free measurement, taken BEFORE these runs (and before the code was read)

`trajectory_ranking_check.py --clip 20`, 54 reachable trajectories (6 seeds × 9 controllers,
2 successful, 14 dock entries):

| reward | `succ>fail` |
|---|---:|
| `L0/cand_02` (original) | **0.038** |
| `f00_copy` (byte-identical copy) | 0.038 — machinery check passes |
| **`f01_align_fix`** (one subscript) | **0.702** |
| `f02_align_fix_units` | 0.423 |
| `f03_align_fix_gentle` | 0.702 |
| `L2/cand_00` (reference) | 0.798 |

(The same file scored 0.268 in the ladder run, which used an 8-seed library — the score is
library-size sensitive, so only the direction is interpretable. Both readings are < 0.5 for
the original: it ranks success *below* failure.)

## Arms (`make_align_fix_variants.py`, edits recorded in each file header)

| arm | edit |
|---|---|
| `f00_copy` | none (byte-identical body) — machinery control |
| `f01_align_fix` | `heading_align`: `1 - abs(obs[10])` → `1 - abs(obs[11])`, nothing else |
| `f02_align_fix_units` | `f01` + `crate_speed` restored to m/s (obs[8]/[9] are ÷3.0, so its speed penalties are 9× too weak) |
| `f03_align_fix_gentle` | `f01` + probe D's obs-only gentleness term |

## Protocol (fixed)

`configs/env007_terminal_rule_pilot.yaml`, `n_envs=6`, clip 20, **1.2M** steps, seed 0;
scored on fresh seeds **32000–32059** (60 episodes) alongside matched re-scores of the two
original 1.2M models, so every comparison is inside one seed block.

## Predictions, fixed before launching

| # | prediction |
|---|---|
| **A1** | `f00_copy` = **0/60** (byte-identical copy; if this fails, treat the run as void) |
| **A2** | `f01_align_fix` **> 0/60** — i.e. the inverted alignment was the operative cause |
| **A3** | `f03 ≥ f01` (gentleness is the term probe D shows is decisive, so it should not hurt) |
| **A4** | if the check's *relative* order means anything, `f02 ≤ f01` (the ranking readout predicts 0.423 < 0.702) |

**A2 is a hypothesis, and a null result is a result.** `SESSION_STATE.md` §3f established
that ranking correctness does not imply learnability (control v1 scores 1.000 and trains to
0 %). So:

* if `f01 > 0/60` → a one-subscript generation error fully explains this arm's failure, and
  the cheap check flagged it beforehand;
* if `f01 = 0/60` → this is a **second, independent instance of §3f**: repairing the reward's
  ordering is not sufficient to make it learnable, and the arm has more than one cause. That
  outcome must be reported as §3f evidence, not as a failed experiment.

## Standing caveat

Oracle-authored edit (as in `REPAIR_TEST_PREREGISTRATION.md`): it bounds what a repair
operator could reach and says nothing about whether a search would find it. Never report it
as a method result.
