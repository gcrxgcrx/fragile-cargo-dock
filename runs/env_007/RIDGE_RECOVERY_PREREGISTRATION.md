# Pre-registration: ridge-recovery operator test (does a loop's *diagnosis* beat a plain edit?)

Design fixed before any run. Purpose: answer "does CREATE have an advantage?" on the axis the
user identified — **who can measure/calibrate the release-lead-time parameters well** — in a way
that is far more sensitive than "can the loop invent a reward from scratch", because **the answer
is known**.

## Why this design

`runs/env_007/APPROACH_ABLATION_PREREGISTRATION.md` + `OVERSHOOT_ABLATION_PREREGISTRATION.md`
established that the working reward is a **narrow ridge**: a specific delta-progress coefficient
(~×50 of the candidate's own term) plus a specific payoff form (a dense per-step payoff on the
instantaneous success predicate). Nine single-axis perturbations of the working reward all fall
to **0/60** against a matched **16/60** baseline in the same fresh block, and two of them have
*diagnosable, distinct signatures*:

| off-ridge seed | its known defect | measured signature |
|---|---|---|
| `p02_progress_x200` (rewrite of the working recipe, coefficient 2400 instead of 600) | too much pull | fresh **0/60**, `dock_entered` **0.72** (arrives, cannot settle), mean return −12.44 |
| `o01_overshoot` (working recipe + a −20/step positional overshoot cliff) | punishes the only available route | fresh **0/60**, `dock_entered` **0.00** (avoidance), mean return −1.47 |

Both have a 1.2M `training_summary.json` on disk, so a diagnosis/reflection can be built from real
training dynamics — and each has a **known-good nearby answer** (the unmodified recipe scores
16/60 in the same block). A loop that can *measure* its way back across the ridge is doing the
calibration the method claims to do; one that cannot is only sampling.

## Operators (both driven by existing code, same model, same temperature)

* **EUREKA edit path** — `build_reward_reflection(train_dir)` → `materialise_reward(mode="edit")`
  with `prompts/eureka_02_reward_edit.md` (exactly the path measured at real 0/8 vs sham 1/8 in
  `REPAIR_LOOP_FINDINGS.md`, now applied to a *known-cause* seed).
* **CREATE revision path** — `pipeline/run_04_build_iteration_context.py --train-run-dir <seed>
  --out <context.md>` (its analysis LLM produces the diagnostic JSON: `failure_modes`,
  `hacking_risks`, `recommended_action`, `component_analysis`, plus matched misalignment cards),
  then `pipeline/run_05_reward_revision.py --previous-reward <seed> --iteration-context
  <context.md>` for the revised reward.

Protocol per arm: `configs/env007_terminal_rule_pilot.yaml`, `n_envs=6`, clip 20, 1.2M steps,
seed 0, **4 repairs per (operator, seed)**, scored on **36000–36059** (a block used for the first
time here; 30000/32000/33000/34000/35000 stay untouched by this experiment).

Cost: 2 operators × 2 seeds × 4 repairs = **16 trainings ≈ 50 min**, plus 16 generations (≈5 min).

## Predictions, fixed now

| # | prediction |
|---|---|
| **R1** | at least one **CREATE-path** repair recovers delivery (**≥ 8/60**, i.e. statistically separable from 0/60) on at least one of the two seeds |
| **R2** | CREATE's recovery rate > EUREKA's (one-sided Fisher over 4 repairs per cell, pooled over the two seeds: 8 vs 8) |
| **R3** | the *specific* recovery shape differs by seed cause: from `p02` (arrives, cannot settle) a successful repair should show `dock_entered` **≥ 0.5 with success > 0**; from `o01` (avoidance) a successful repair should show `dock_entered` **rising from 0.00** |
| **R4** | EUREKA's edit path mostly returns near-identical code (its measured behaviour: no diagnosis, only the reflection tables) — i.e. its repairs stay 0/60 |

## Interpretation, fixed now

* **R1 + R2 hold** → CREATE's structured diagnosis has a measurable advantage, **on the
  calibration axis**, with a known-correct answer. That is the first positive method-difference
  evidence available on this environment, and it justifies the full-pipeline run (costs in
  `runs/env_007/COMPUTE_LEDGER.md`: ≈1.5 h EUREKA, ≈5 h CREATE at 3M — and ×3 if the config's
  `multi_seed.num_seeds: 3` default is kept, which MUST be pinned first).
* **R1 fails** → neither operator can cross the ridge even when the defect is diagnosable and the
  target is known to exist. Then the honest conclusion is that the ceiling is the operator's
  *implemented* calibration ability, and running the full pipelines would be spending 6–14 h on a
  loop whose repair step is measurably inert. Report that, do not run them.
* Either way, this does **not** re-open the channel question (real vs sham, already measured 0/8
  vs 1/8); it tests the *operator* with the evidence held fixed at "real".

## Standing caveats

* Oracle-authored *seeds*: the two off-ridge candidates were built by me. That is the point —
  it makes the defect known — but it means the test measures recovery from *these* defects, not
  from arbitrary ones.
* 4 repairs per cell is small; the pre-declared bar (≥ 8/60 on one seed, and a pooled Fisher
  between operators) is the mitigation.
* One round only. If a single revision cannot recover the ridge, a multi-round loop might still
  (untested); that distinction must be stated with any negative result.
