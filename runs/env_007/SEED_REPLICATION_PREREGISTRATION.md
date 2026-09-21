# PRE-REGISTRATION: seed-budgeted replication of the close-speed isolation

Written before the new seeds were trained (05:10). Question: the isolation experiment
(`runs/env_007/CLOSE_SPEED_ISOLATION_FINDINGS.md`) gave a clean mechanism — the added term
moves max consecutive stable steps from 0.83 to 5-8 and seed 0 from 0/60 to 46/60 — but the
comparison **reversed on seed 1** (48/60 vs 26/60) and pooled p = 0.55. With one seed per arm
that is unanswerable. This experiment buys the seeds.

## Arms and seeds

`C0` = `runs/env_007/close_speed_test/C0_control/reward_v1.py` (byte-equivalent to
`control_v1`); `C1` = `.../C1_closing_speed/reward_v1.py` (adds one observation-only
closing-speed penalty; verified equal to `probeD`).

**Training seeds 0, 1, 2, 3 for both arms** (8 runs). Seeds 0 and 1 already exist for both arms
and are reused (verified equivalent: `C0`/0 = `control_obs_only/train_1m2`, `C0`/1 and `C1`/1
new in this round, `C1`/0 = `ablation_probe/probeD`). Protocol unchanged: 1.2 M steps,
`n_envs = 6`, clip 20, `normalize_reward = true`, gamma = 0.999.

**Evaluation: fresh block `38000-38059` for every arm and seed** — one block, so that all
paired comparisons sit inside it. (Block 37000-37059 was used by the previous wave; using one
block for all 8 runs is what makes the paired differences interpretable.)

## Primary analysis, fixed now

Unit of analysis = **training seed**, n = 4 per arm. For each seed: success/60 and
`dock_entered`. Reported:

1. **paired per-seed success difference** `d_i = C1_i - C0_i` for i = 0..3, its mean and
   standard deviation, and a **paired t-interval** (n = 4, two-sided 95 %);
2. the per-arm success distribution (mean, sd, min, max) and `dock_entered` distribution;
3. the mechanism statistic — **max consecutive stable steps** per arm, from
   `diagnose_settling.py`.

## Decision rule, fixed now (this replaces S5 from the previous pre-registration)

* **P (pass)**: the 95 % paired interval for the mean success difference **excludes 0** AND the
  mean difference is positive. -> the isolated term is established at the seed level, and a
  pipeline run becomes defensible.
* **N (null)**: the interval contains 0. -> the isolation effect is **not** established at the
  seed level; the honest statement becomes "the mechanism is real (settling statistic), but its
  effect on success is within seed noise at 1.2 M", and a pipeline run is **not** justified by
  it.
* Either way the **mechanism** claim is reported separately and is not affected by this rule:
  it rests on the max-stable-step contrast, which was measured on the same block.

## Addendum experiment (added 06:24, before it ran): is the seed class stable across budget?

The main experiment's §2b found, for `C0`, that a bad 1.2 M seed is still bad at 3 M (0 -> 2/60)
and a good one is still good (48 -> 51/60). That was two seeds of one reward. This addendum
repeats it on the other arm, with the same block and protocol:

| run | reward | 1.2 M (block 38000-38059) | 3 M |
|---|---|---:|---|
| `C1` seed 0 | `C1_closing_speed` | 45/60 | (running) |
| `C1` seed 2 | `C1_closing_speed` | 0/60, dock 0.02 | (running) |
| `C0` seed 0 | `C0_control` | 0/60 | **2/60** (measured) |
| `C0` seed 1 | `C0_control` | 48/60 | **51/60** (measured) |

**Prediction, fixed now:** the 1.2 M score predicts the 3 M class for `C1` too, i.e.
seed 0 stays clearly non-zero (>= 20/60) and seed 2 stays near zero (<= 8/60).

**If it holds on both rewards**, then a single 1.2 M run is a **valid rejection filter**
(a near-zero 1.2 M candidate does not become a success at 3 M) while remaining **invalid for
selection** — which is the concrete design rule the pipeline needs, and it is cheap.

**If it fails** (a 0/60 seed at 1.2 M becomes good at 3 M), then no cheap screen exists and the
pipeline must budget >= 3 training seeds per candidate.


* It can settle whether the 0 % -> 77 % single-seed contrast survives replication, which is the
  question the pipeline's selection needs answered before it is worth 1.5-5 h.
* It **cannot** establish a *general* seed-level effect size: n = 4 per arm gives a wide
  interval, and one arm's seeds come from a distribution whose spread we have already seen to
  be large ({0, 48} for the control on two seeds). A null here therefore means "not
  established", not "no effect".
* It does not touch the v8 question. `v8_cand_01`'s 4/60 (block 38000-38059) is reported
  alongside for context only; it is a different block from the pairs? — no: **38000-38059 is
  the same block**, so the v8 number is directly comparable and is included as a reference row.
