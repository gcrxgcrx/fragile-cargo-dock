# SESSION REPORT — 05:05, the autonomous round (isolation → v8 → calibration)

Everything below is measured in this session; raw JSON, protocols and predictions are in the
files named next to each result. Three experiments were run; one of them falsified my own
hypothesis. Nothing here re-litigates earlier results.

## 1. Isolation of the one unexplained contrast — DONE, clean result with an honest caveat

`runs/env_007/CLOSE_SPEED_ISOLATION_{PREREGISTRATION,FINDINGS}.md`; arms built by
`make_closing_speed_variants.py`; evaluation and mechanism readout in
`close_speed_test/{eval_block37000_wave1,eval_block37000_wave2,settling_block37000_wave1}.json`.

`C0` = `control_v1`; `C1` = `C0` + one observation-only closing-speed term
(`max |C1 - probeD| = 0`, `max |C0 - control_v1| = 0` over 27 trajectories). 1.2 M steps,
paired training seeds, one fresh block (37000-37059):

| arm | seed | success | dock | P(success\|dock) | max consecutive stable steps |
|---|---:|---:|---:|---:|---|
| C0 | 0 | **0/60** | 0.90 | 0.000 | **0.83** (median 1) |
| C0 | 1 | **48/60** | 0.87 | 0.923 | — |
| C1 | 0 | **46/60** | 0.87 | 0.885 | **8.03** (median 10) |
| C1 | 1 | **26/60** | 0.70 | 0.619 | **4.93** (median 5) |
| C1 | 2 | **0/60** | 0.13 | 0.000 | — |

**What it establishes.** `C0` reaches the dock in 90 % of episodes and never holds the settled
condition for even one step on average; 0 of its 54 docked-and-failed episodes got within
three steps of success. The isolated term moves that statistic to 5-8 steps. The missing
capability of the failing family is therefore **settling**, and the observation contract can
express it.

**What it does not establish.** Seed 1 reverses the comparison (48 vs 26 in the control's
favour) and the pooled difference over 120 episodes is **Fisher p = 0.55**. At 1.2 M steps with
one training seed per arm, seed variance is comparable to a large single-variable effect.

## 2. The v8 prompt revision — form fixed, ceiling unchanged

`make_prompt_v8.py` → `prompts/eureka_01_initial_reward_v8.md` (+ `runs/env_007/v8_prompt.diff`),
predictions and verdicts in `runs/env_007/V8_PROMPT_PREREGISTRATION.md`.

Three auditable edits to v7: the one-off event is demoted to optional with the clip arithmetic
stated; the settled per-step payoff must not be switched off by the candidate's own event; the
"all other components must be exactly 0 when settled" rule is narrowed to persistent *state*
bonuses, and progress must be signed.

* 8/8 candidates valid; **7/8 pay per step on a settled state after their own event fired**
  (v7 family: 0-1/8).
* `cand_01` trained (1.2 M, seed 0), scored on fresh block 38000-38059: **4/60 = 6.7 %**,
  `dock_entered` **0.70**. Its **progress term is bounded at 14.04 points/episode**, on the
  measurement where `v7_cand_01` accumulated **2747**, and its settled payoff is now the
  dominant component (share 0.761).
* **It still does not settle**: max consecutive stable steps **2.42** (median 1).

This is the best unedited LLM candidate the project has produced (previous best 3/60 with
`dock_entered` 0.18), and it is still an order of magnitude below the hand-written arms.

## 3. The calibration hypothesis — my own prediction, falsified

`runs/env_007/CALIBRATION_TEST_PREREGISTRATION.md`; variants by `make_calibration_variant.py`.
The reading being tested was: the LLM writes the right term but with too small a coefficient,
so scale that one literal and nothing else.

| arm | coefficient on `contact*closing` | success | dock | return |
|---|---:|---:|---:|---:|
| `v8_cand_01` | −1.0 | **4/60** | **0.70** | 26.5 |
| `gx15` | −15.0 | **0/60** | **0.00** | −1.3 |
| `gx50` | −50.0 | **0/60** | **0.00** | −1.2 |

Scaling it 15x does not make the term effective; it makes the optimum **not to approach**
(immobility, return ≈ −1.3 — the same signature as the nine earlier guidance ablations and as
`r08`: adding gentleness to a candidate that already penalises closing speed destroys
locomotion). The hypothesis is falsified, and the same scalar is demonstrably not transferable
between reward contexts.

## 4. Where the project stands after this round

Four candidate explanations for the LLM-vs-hand-written gap have now been tested and closed:
structure/prompt shape (§2), information/evidence channel (real 0/8 vs sham 1/8),
reachability (C0 docks 90 %), and term magnitude (§3). What remains, and what the data point
at, is **seed-level optimisation variance / attractor selection** — the same phenomenon that
gives the x100 coefficient {31, 43, **0**} out of 60 across three seeds, and that gives the
control {0, 48} across two.

Consequence for the full pipeline: the pre-registered S5 rule is met only literally. Under the
bar actually stated for entering the pipeline — a candidate at **>= 20/60 with the effect
visible on >= 2 independent training seeds** — we are **not there**: the best arm is 48/60 but
that is the *control*, and the best LLM candidate is 4/60 on one seed. A pipeline run now would
cost 1.5-5 h and most likely reproduce "both 0/20", which we already have.

## 5. What the next experiment should be (proposal, not yet run)

Because the surviving explanation is optimisation variance, the next step is not another
prompt revision. The two candidates worth costing out:

1. **Seed-budgeted replication of the best arm.** `C1`/seed 0 (= `probeD`) already scores
   46-48/60 on two different fresh blocks. Train 4 more seeds of `C0` and `C1` (8 runs, ~25 min
   in two parallel waves) and report, per arm, the distribution of success and of
   `max consecutive stable steps`. This settles whether the isolation effect is real at the
   seed level — the question this round could not answer — and it is exactly the measurement
   the pipeline's selection needs.
2. **A multi-round operator test** (never run): seed the loop with a candidate whose defect is
   known and see whether one round of diagnosis+edit recovers it
   (`runs/env_007/RIDGE_RECOVERY_PREREGISTRATION.md`).

My recommendation is 1, then 2, then — only with a >= 20/60 candidate visible on two seeds —
the pipeline.
