# PRE-REGISTRATION: isolating the one term that separates 0 % from 73.3 %

> **STATUS: COMPLETE — see `runs/env_007/CLOSE_SPEED_ISOLATION_FINDINGS.md`.** Outcome: the
> mechanism is clean (settling: 0.83 -> 8.03 max consecutive stable steps; 0/60 -> 46/60 on
> seed 0), the pre-registered S5 rule is met literally, but **seed 1 reverses the comparison**
> (48/60 vs 26/60) and the pooled difference is p = 0.55.

Written before any training of the new arms. Question: `control_v1` (`C0`) scores **0/60**
and `probeD` scores **73.3 %**. Their source files differ by **exactly one term** — an
observation-only closing-speed penalty:

```python
cos_h, sin_h = obs[2], obs[3]
cart_forward_speed = obs[4] * 3.0
crate_speed_along_heading = cvx * cos_h + cvy * sin_h
closing_speed = cart_forward_speed - crate_speed_along_heading
if closing_speed < 0.0:
    closing_speed = 0.0
contact = 1.0 if next_obs[14] > 0.5 else 0.0
gentleness = -0.05 * contact * closing_speed
```

Everything else — the two telescoping shaping terms, the `+20` settled-state hold, the
boundary guard, the shove penalty, all coefficients, the training budget and the harness —
is identical, verified by diffing the two files.

This is the highest-information single-variable experiment the project has left: it is the
only measured contrast in which a *single added term* moves an arm from 0 % to 73.3 %, and
it has never been isolated against a matched baseline in the same seed block.

**Note on arm identity.** `probeD`'s realised component table
(`runs/env_007/ablation_probe/probeD/component_stats.md`) is exactly `control_v1` plus
`gentleness_obs` (mean −0.000752, nonzero rate 0.1455). So `probeD` **is** `C1`, and its
seed-0 model is reused rather than retrained.

## Arms

| arm | reward file | difference from `C0` |
|---|---|---|
| **`C0`** | `runs/env_007/close_speed_test/C0_control/reward_v1.py` | — (byte-identical body to `runs/env_007/control_obs_only/reward.py`) |
| **`C1`** | `runs/env_007/close_speed_test/C1_closing_speed/reward_v1.py` | `+` the closing-speed term above, nothing else |

Built by `make_closing_speed_variants.py` with auditable anchor checks, exactly like
`make_repair_variants.py`.

## Protocol

`configs/env007_terminal_rule_pilot.yaml` (n_envs = 6, `reward_clip` = 20,
`normalize_reward` = true, γ = 0.999), **1.2 M** steps — identical to every other
measurement in this project — for **training seeds 0, 1, 2** and both arms (6 runs;
`C0`/seed 0 and `C1`/seed 0 are already on disk and are re-scored, not retrained).

**Evaluation: fresh block `37000–37059`** (60 episodes), used for the first time here, so
that 30000/32000/33000/34000/35000/36000 stay untouched by this experiment. **All six runs
are scored on this same block**, because §5.8 requires comparisons to sit inside one block.

## Predictions, fixed now

| # | prediction |
|---|---|
| **P1** | `C1` beats `C0` on this block, i.e. `C1`'s 60-episode success is non-zero while `C0`'s is 0–1/60 |
| **P2** | the effect is mostly on **settling**, not on reaching: `C1`'s `dock_entered` is high and its `P(success \| dock_entered)` exceeds `C0`'s |
| **P3** | `C0`'s in-training component table shows the settled hold essentially never reached (`dock_settled_hold` active rate ≤ 0.01), while `C1`'s is higher |
| **P4** | whatever the effect size, `C1`'s three seeds are **not** all alike (the ×100 precedent: {31, 43, 0} out of 60) |

## Verdict rule, fixed now

* **S5 (the external "30 % bar")**: `C1` is non-zero on **≥ 2 of 3** training seeds **and**
  at least one `C1` seed is **≥ 8/60** (statistically separable from 0/60, one-sided
  Fisher p < 0.05). *Only if S5 holds is a full-pipeline run warrantable* — the bar is
  that the mechanism we would take into the pipeline has been seen to produce a
  non-noise delivery rate on more than one training seed.
* **S4 (weaker)**: `C1` is non-zero on one seed only, or the paired difference runs in the
  right direction in 2/3 pairs without any seed reaching 8/60.
* **S3/B (negative)**: `C0` and `C1` are indistinguishable on this block, or the difference
  reverses. Then the 0 % → 73.3 % contrast was a **seed lottery** (it was measured with one
  training seed per arm — the same weakness §5.8 flags), and the "missing ingredient"
  question returns to the general form of Q1 with no isolated candidate.
* **Uninformative branch, declared in advance**: if `C1` docks in ≥ 90 % of episodes on
  every seed and still scores < 8/60, the bottleneck is **settling and nothing else** —
  a result that is *also* decisive, and points at `P(settle | reach)` rather than at the
  shaping terms.

## Secondary instrumentation (all cheap, run regardless of the verdict)

For every arm and seed: `success`, `dock_entered`, `P(success | dock_entered)`, mean
episode length, and the in-training component activation rates — so that "reach" and
"settle" are always reported separately (§5.8), and so that the ×100-style
"docks-95 %-and-scores-0" signature can be recognised immediately if it appears.

The `ret_rms.std` / GAE-advantage instrumentation suggested by the external review is NOT
part of this experiment: it needs a training-harness change, and it is only interpretable
once a *pair* of arms is known to differ.
