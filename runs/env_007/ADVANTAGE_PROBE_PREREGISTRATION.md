# Pre-registration: a training-free "advantage-scale" selector

Written 20:22, **before the probe was run on any candidate**. Pool and thresholds are fixed
here; the 14 candidates' outcomes were already known (they are the labels), but no
advantage-scale number had been computed for any of them.

## The statistic, frozen

Library: the same scripted controllers as `trajectory_ranking_check.py`
(6 seeds × 9 controllers, seed offset 50000), harness clip 20.

    per_step(traj) = clipped_generated_return(traj) / len(traj)
    A = mean(per_step over the seven heuristic controllers
             {push_forever, release_0.05, 0.15, 0.25, 0.35, 0.50, 0.80})
        − mean(per_step over idle)

`shove` is excluded from the acting mean (a deliberately bad controller that drives out of
bounds). Per-step normalisation is required because success terminates episodes early.

## Why this statistic — the three measurements that motivate it

1. `L2/cand_00`'s reward orders success (+56) above pushing (+6.4) above idle (−0.8) per
   episode on these trajectories, yet its trained policy sits at ≈−0.07 per episode and
   never moves the crate: ordering is right, the per-step advantage of acting is only
   ≈0.018 (`trajectory_ranking_check.py --describe`).
2. Multiplying **one coefficient** of that same reward by 50 (arm `r04`, nothing added)
   turns a never-moving policy into one that enters the dock in **97 %** of episodes — the
   advantage rises ~50×, the task-directed behaviour appears.
3. Every working hand-written arm carries its mass in a term worth +20…+300 per episode at
   0.03–3 % activation (orders of magnitude larger per step than the LLM rewards' shaping).

Hypothesis: **learnability here is governed by the per-step advantage of task-directed
behaviour over inaction** — a quantity computable on reachable trajectories with no training
at all, and one that neither the structural checks (`analyze_terminal_dominance.py`) nor the
ordering check (`trajectory_ranking_check.py`) measures.

## Pool (fixed)

`runs/env_007/advantage_pool14.json` — the same 14 candidates as the earlier component-activity
readout, so the two instruments are directly comparable:

* **positive (5)**: `probeA` 43.3 %, `probeB` 36.7 %, `probeC` 98.3 %, `probeD` 73.3 %,
  `probeE` 76.7 % — 1.2M outcomes on seeds 30000–30059 (`SESSION_STATE.md` §3b), replicated
  on 32000–32059;
* **negative (9)**: `control_v1` 0 % plus all eight ladder candidates 0/60.

Prevalence = 5/14 = 0.357; predict-all baseline is precision 0.357, recall 1.0, AUPRC 0.357.

## Bars, fixed

The probe is adopted as a training-free selector **iff** `AUPRC >= 0.60` **and** there is a
threshold `t` with `A >= t` giving `precision >= 0.60` and `recall >= 0.60`. (The threshold
is chosen post hoc on this calibration pool and reported as such; the *bars* are fixed now.)

## Predictions

| # | prediction |
|---|---|
| **V1** | the five hand-written probes have larger `A` than every LLM candidate and than `control_v1` |
| **V2** | `control_v1` has the smallest `A` of the pool (it is 0 % *and* has no payoff term at all) — or, if not the smallest, is clearly below the probes |
| **V3** | AUPRC ≥ 0.60 and a qualifying threshold exists → the probe is the first training-free selector found on this environment |
| **V4** *(mechanism, tested on the repair arms as a second pool)* | `A(r04) >> A(r00) ≈ A(r01)`: the ×50 coefficient raises the measured advantage, which is why the policy moves. If this holds, the probe tracks the mechanism causally in a case where **only one number changed** |

## Falsification

* If `AUPRC <= prevalence` (0.357), the advantage scale is not a usable selector and V3 fails.
* If `A(r04) ≈ A(r00)`, the probe is not measuring what the ×50 experiment changed, and V4's
  causal claim fails regardless of V3.

## Honest limitation, stated in advance

`A` measures whether a reward makes *task-directed behaviour* profitable, not whether it
makes *task success* profitable. Arm `r04` is expected to score **high** while being 0/60
(it docks in 97 % of episodes but cannot settle), so on a pool that included it as a negative
the probe would produce a **predicted false positive**. That is a property of the statistic,
not a surprise to be discovered later, and it must be reported with any positive result.
