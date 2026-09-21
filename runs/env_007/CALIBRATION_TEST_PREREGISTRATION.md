# PRE-REGISTRATION: is the missing ingredient the *effect size* of one term?

Written before the two calibration arms were trained (04:55).

## The reading being tested

`runs/env_007/CLOSE_SPEED_ISOLATION_FINDINGS.md` established that the one term separating
0/60 from 46/60 is an observation-only closing-speed penalty, and that it acts on **settling**
(max consecutive stable steps 0.83 -> 8.03), not on reaching the dock.

`runs/env_007/V8_PROMPT_PREREGISTRATION.md` reports the first LLM candidate whose reward
*shape* is right: `v8_cand_01` has a bounded signed progress term (14.04 points/episode, where
`v7_cand_01` accumulated 2747), a dominant per-step settled payoff (share 0.761) that is not
switched off — and still only **4/60**, with **max consecutive stable steps 2.42**.

That candidate's gentleness term *exists* but is worth **-1.73 points/episode**; in the
hand-written arm the same functional form is what produces the 8-step hold.

**Hypothesis:** the binding constraint is no longer the reward's structure but the **effect
size** of the term that governs settling. If true, scaling *only* that coefficient must move
`max_stable`.

## Arms

`make_calibration_variant.py` rewrites exactly one literal in `v8_cand_01`'s source:

| arm | edit | nothing else |
|---|---|---|
| `gx15` | `gentleness = -1.0 * contact * closing` -> `-15.0` | unchanged |
| `gx50` | -> `-50.0` | unchanged |

Both: 1.2 M steps, seed 0, `configs/env007_terminal_rule_pilot.yaml`, evaluated on the fresh
block **38000-38059** (same block as `v8_cand_01`'s 4/60, so the comparison is inside one
block).

## Predictions, fixed now

| # | prediction |
|---|---|
| **C1** | at least one calibration arm reaches **>= 8/60**, i.e. above the prompt-only candidate and separable from 0/60 |
| **C2** | its **max consecutive stable steps** rises well above 2.42 (towards the hand-written arm's 8.0) |
| **C3** | `dock_entered` does **not** fall below 0.5 — the isolation experiment showed gentleness did not cost reachability (0.87 vs 0.90), and the earlier `r08` failure (0/60, dock 0.00) came from stacking a second closing-speed penalty, which this is not |
| **C4** | if `C1` fails while `C2` holds, the ceiling is not settling either, and the remaining explanation is the optimisation/seed variance measured in `CLOSE_SPEED_ISOLATION_FINDINGS.md` |

## STATUS: COMPLETE — both predictions FAILED, hypothesis falsified (scored 05:06)

Both arms trained (1.2 M, seed 0) and evaluated on block **38000-38059**, the same block as
`v8_cand_01`'s 4/60:

| arm | coefficient on `contact*closing` | success | dock | mean return |
|---|---:|---:|---:|---:|
| `v8_cand_01` (unmodified) | −1.0 | **4/60** | **0.70** | 26.48 |
| `gx15` | −15.0 | **0/60** | **0.00** | −1.30 |
| `gx50` | −50.0 | **0/60** | **0.00** | −1.23 |

* **C1 FAILS** (no arm reaches 8/60) and **C2 FAILS** (`dock_entered` collapses to 0.00, so
  there is no settling to measure).
* The failure mode is **immobility**, not a settling shortfall: the return is ≈ −1.3, i.e. the
  policy sits still, exactly the signature of the nine earlier guidance ablations
  (`OVERSHOOT_/APPROACH_ABLATION_PREREGISTRATION.md`: adding gentleness to a candidate that
  already penalises closing speed destroys locomotion — `r08` 0/60 dock 0.00, `r10` 0/60).
  Scaling the coefficient 15x does not make the term *effective*; it makes the optimum
  **not to approach**.
* So the "the LLM writes the right term but too small a coefficient" hypothesis is
  **falsified**: bringing the coefficient to the hand-written magnitude breaks the behaviour
  rather than fixing it. The hand-written arm works at a *particular* value in a *particular*
  reward context, and the same scalar is not transferable across contexts.

### What survives

The three measured facts that stand after this round:

1. the missing *capability* for the failing family is settling, and one observation-only term
   supplies it in a controlled single-variable test (`0.83 -> 8.03` max consecutive stable
   steps; 0/60 -> 46/60 on seed 0) — but the same comparison **reverses on seed 1**
   (48/60 vs 26/60) and is not significant pooled (p = 0.55);
2. ridding the LLM candidate of its structural defects is possible and measurable
   (`v8_cand_01`: bounded progress 14.04 vs 2747, settled payoff dominant, payoff not switched
   off) and moves it from 0-3/60 to **4/60 with `dock_entered` 0.70** — the best unedited LLM
   candidate this project has produced — while **still not settling** (2.42 steps);
3. that candidate's gentleness coefficient cannot simply be scaled up (this experiment).

**The remaining explanation for the LLM-vs-hand-written gap, after ruling out shape (§V8),
information (§3.4 of the handoff), reachability and magnitude (this experiment), is
seed-level optimisation variance / attractor selection** — the same phenomenon that makes the
x100 coefficient score {31, 43, 0} out of 60 on three seeds. That is a property of the
optimisation, not of the reward text, and it is where the next experiment should aim.


* **C1 and C2 hold** -> the binding constraint is the **calibration** of the settling term.
  Then the honest next experiment is an explicit coefficient-calibration stage (a small sweep
  over that one scalar), *not* a full pipeline run, and the v8 prompt should state the
  required **effect size** ("the term must dominate the per-step budget when closing fast"),
  not just the functional form.
* **C2 holds, C1 fails** -> the policy learns to settle but the last step (the 10-step hold)
  is still rare; look at `P(settle | reach)` and at what breaks the streak.
* **Both fail** -> the calibration hypothesis is falsified, the v8 candidate's failure is not
  a scale problem, and the remaining candidates are optimisation variance or a missing
  signal the observation contract cannot express.

## Standing caveats

* One training seed per arm. `CLOSE_SPEED_ISOLATION_FINDINGS.md` §4 measured that seed
  variance at 1.2 M is comparable to a large single-variable effect, so a *positive* result
  here is weaker than it looks and a *negative* one cannot exclude the effect. This is
  stated in advance because it is exactly the mistake the previous experiment caught.
* Two arms only, chosen as 15x and 50x: the earlier ridge sweep showed the scale axis is
  **non-monotone** (x50 -> 16/60, x75 -> 0-4/60, x100 -> 31-43/60, x200 -> 0/60), so a null
  result at two points would not prove the axis is flat.
