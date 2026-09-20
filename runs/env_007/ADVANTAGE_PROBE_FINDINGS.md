# ADVANTAGE-SCALE PROBE FINDINGS — a good diagnostic that is not a selector

Pre-registration (statistic, pool, bars, predictions): `runs/env_007/ADVANTAGE_PROBE_PREREGISTRATION.md`,
written before the probe was run on any candidate. Tool: `advantage_scale_probe.py`.
Raw: `runs/env_007/advantage_probe_pool14.json`, `advantage_probe_repair.json`.

Statistic: on the scripted-trajectory library used by `trajectory_ranking_check.py`
(6 seeds × 9 controllers, seed offset 50000, clip 20),

```
A = mean per-step generated return over the seven heuristic controllers
    − mean per-step generated return over `idle`
```

## 1. As a cross-candidate selector it is REJECTED

`advantage_pool14.json` (5 known-positive hand-written arms, 9 known-negative LLM arms and
the control):

| candidate | label | A (per step) |
|---|---:|---:|
| `L0_cand_13` | 0 | **+0.394** |
| `L0_cand_02` | 0 | **+0.315** |
| `L0_cand_11` | 0 | **+0.173** |
| `probeA` | 1 | +0.044 |
| `control_v1` | 0 | +0.044 |
| `probeD` | 1 | +0.041 |
| `probeE` | 1 | +0.017 |
| `probeB` | 1 | +0.012 |
| `probeC` | 1 | +0.012 |
| `L2_cand_14` | 0 | −0.000 |
| `L0_cand_00` | 0 | −0.139 |
| `L2_cand_04` | 0 | −0.250 |
| `L2_cand_00` | 0 | −0.259 |
| `L2_cand_05` | 0 | −0.343 |

`AUPRC = 0.413` against a prevalence baseline of `0.357`; the best threshold with
recall ≥ 0.60 gives precision 0.556. **V1, V2 and V3 are falsified**, and in the worst
direction: the three *failing* L0 candidates have the **largest** advantage of the pool,
above every working arm. The reason is plain in hindsight — a dense farmable term
(`crate_docking_quality`, active every step, +311 per episode) pays *more* while acting than
while idling, so an advantage scale cannot distinguish "paid to do the task" from "paid to
do anything".

## 2. Within a family it is a *causal* diagnostic — V4 confirmed

`advantage_pool_repair.json` (the eleven repair variants; all 0/60, so the metric part of the
report is vacuous and the **A column is the point**):

| arm | edit | A (per step) |
|---|---|---:|
| `r09_progress_x50_stream` | progress ×50 + payoff stream | **+0.141** |
| `r10_x50_stream_gentle` | progress ×50 + stream + gentleness | **+0.138** |
| `r04_own_progress_x50` | progress ×50 only | **+0.113** |
| `r08_progress_x50_gentle` | progress ×50 + gentleness | **+0.111** |
| `r07_stream_and_guidance` | guidance + stream | −0.223 |
| `r03_guidance_x3` | guidance ×3 | −0.223 |
| `r06_settled_stream` | payoff stream only | −0.232 |
| `r01_guidance` | guidance only | −0.247 |
| `r00_copy` | base candidate | −0.259 |
| **`r11_x50_own_speedpen_x100`** | progress ×50 + own speed penalty ×100 | **−12.825** |

Computed with **no training**, this predicts exactly which arms move the cart — the four
containing the ×50 change flip `A` from −0.259 to +0.11…+0.14, and those are the four whose
trained policies dock (measured: `r04` enters the dock in 97 % of episodes, goal distance
4.148 m → 0.140 m, while every non-×50 arm stays at exactly 4.148 m). The probe also flags
`r11` as catastrophic (`A = −12.8`): scaling the candidate's **own** speed penalty by 100
makes acting ruinously unprofitable, so the prediction is that `r11` is *less* mobile than
the base — recorded here before `r11`'s result was read.

## 3. What this means — the criterion is two-part

`A > 0` is **necessary**: if acting is worse than idling, the reward's optimum *is* inaction,
and PPO's apparent "failure to learn" is actually correct optimisation. That is exactly what
the whole L2 family is: `A` = −0.0004 … −0.343, and their trained policies sit still
(goal distance unchanged at 4.148 m, `success_event` activation 0). It is also why a single
coefficient (×50 on the candidate's own progress term) is enough to start moving.

`A > 0` is **not sufficient**: `L0_cand_11/02/13` reach `A` up to +0.39 and are still 0/60,
and `r04` reaches +0.113, docks in 97 % of episodes, and is still 0/60 because settling is
unprofitable. In those cases the reward pays, but for the **wrong behaviour**.

So learnability on this environment requires both

1. **`A > 0`** — acting beats inaction, and
2. **the reward's argmax over reachable behaviour is the success behaviour**

and every instrument tried in this project measures at most one of them:

| instrument | measures | outcome |
|---|---|---|
| `analyze_terminal_dominance.py` (structural) | term presence / ratios on probe states | no correlation with success (`LADDER_FINDINGS.md` §1) |
| `trajectory_ranking_check.py` (ordering) | whether success outranks failure on reachable trajectories | separates and is still useless (§3f; and now `f01_align_fix`: ordering repaired 0.038 → 0.702, still 0/60) |
| component-activity readout (1.2 M) | which term dominates the realised reward | 14/14 in-sample, fails at 0.6 M and at 1.0 M (`LADDER_FINDINGS.md` §5) |
| **advantage scale (`A`)** | **condition 1** | rejects as a selector, but correctly predicts inaction vs action without training |

Condition 2 remains unmeasured by anything. That is the sharpest statement this session can
make about the selection problem, and it is consistent with the project's standing reading
that the bottleneck is generation, not selection.
