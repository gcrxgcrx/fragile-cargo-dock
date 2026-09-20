# Pre-registration: does ADDING an approach reward help? (and in which form?)

Written before any arm was trained. Second question from the user's domain experience:
*"那我们给接近泊位加奖励呢"* — instead of penalising overshoot, reward getting close.

**This is not a side idea: it is already the measured fix.** The two-edit repair that took two
independent candidates from 0/60 to 23/60 has as its **first** edit a much stronger approach
reward — the candidate's own `crate_to_dock_progress` coefficient ×50. The open questions are
therefore about **form** and **size**, not about whether approach reward matters.

## Base and shared baseline

Base: `runs/env_007/repair_test/r09_progress_x50_stream/reward_v1.py` (23/60, dock 0.98).
Shared matched baseline: `runs/env_007/overshoot_ablation/o00_base/` — a byte-identical copy of
it, trained and scored on the **same** fresh block **35000–35059**. No cross-block comparison is
used anywhere in this experiment (block drift is ±12 points; `LADDER_FINDINGS.md` §4).

## The two forms, and why the distinction is the experiment

| form | what it pays | measured precedent |
|---|---|---|
| **delta / progress** | the *reduction* in distance (`d_now − d_next`, gated) | edit #1 of the 23/60 repair; `r04` (this edit alone): dock 0.97 but **0/60** |
| **state / proximity** | *being* close, regardless of how it got there | the `L0` family's `crate_docking_quality`: **100 % active, +311 per episode, 0/60** — the hover-farming pathology |

## A design error the advantage probe caught before training (kept in the record)

The first version of the `p01`/`p04`/`p05` blocks used `1/(1+30·d)`. `obs[12]/[13]` are
normalised by `FIELD_HALF_W/H` (5 m, 4 m) and the **spawn** distance is ≈0.82 in those units,
so that term was a near-constant ≈+0.78 **per step across the whole arena**. The probe's
`idle` column exposed it: the do-nothing controller earned **+0.73/step**. It was not a
proximity reward at all. Fixed to `20·exp(−d/0.05)`, after which `idle` = **0.000** for every
arm. This is recorded because it is a good example of what the `per_step_idle` readout is for.

## Arms (5, one 1.2M run each, seed 0; scored on 35000–35059)

| arm | edit relative to `r09` |
|---|---|
| `p01_proximity` | + dense **state** reward for being near: `20·exp(−d/0.05)` |
| `p02_progress_x200` | the **delta** term 600 → 2400 (i.e. 200× the original 12) |
| `p03_x200_gentle` | `p02` + closing-speed penalty |
| `p04_proximity_gentle` | `p01` + closing-speed penalty |
| `p05_funnel` | + dense reward for being near **and slow**: `20·exp(−d/0.05)·max(0, 1 − v/0.3)` |

(`p05` is the strongest form of the user's idea: a smooth gradient into the settled state, paid
only when close *and* slow — i.e. the release-lead-time skill expressed continuously.)

## Training-free predictions, computed before training

| arm | `A` (per step) | acting | idle |
|---|---:|---:|---:|
| `p02_progress_x200` | **+1.281** | 1.281 | 0.000 |
| `p03_x200_gentle` | +1.278 | — | 0.000 |
| `p01_proximity` | **+1.162** | 1.162 | 0.000 |
| `p04_proximity_gentle` | +1.159 | — | 0.000 |
| `p05_funnel` | +0.517 | 0.517 | 0.000 |
| *`o00_base` (baseline)* | *+0.141* | — | *0.000* |

Every approach arm raises `A` well above the working baseline, so all of them should at least
*act*. `A` says nothing about whether the resulting behaviour *completes*.

## Predictions

| # | prediction |
|---|---|
| **Q1** | `p02 > o00_base` **or** `p02 < o00_base`: registered two-sided. More delta pull either helps (the ×50 ceiling was not the constraint) or tips into overshoot/violent contact. The direction is the result |
| **Q2** | `p01 ≤ o00_base`: the **state** form is the one with a measured pathology (the `L0` family farmed a 100 %-active quality term to +311/episode and 0/60). Mechanism check: if `p01` shows a ~100 %-active proximity term in its component table **and** high `dock_entered` with 0 success, that is farming |
| **Q3** | `p05 ≥ o00_base`: the smooth funnel is the strongest form of the idea and should not hurt |
| **Q4** | if Q1 shows `p02 < o00_base`, then `p03 > p02` — the closing-speed penalty recovers what extra pull costs, i.e. the lead-time skill is what binds once the pull is strong |
| **Q5** | mechanism, from the summaries: in any arm that fails while `dock_entered > 0`, the dominant term should be the *dense* one (proximity/funnel) rather than the settled payoff |

## Interpretation, fixed now

* **Q3 holds and Q2 fails** → the useful way to "reward approaching" is a **dense funnel that
  pays only when close and slow** (a continuous version of the settled payoff), not a state
  reward for proximity. That is the form to add to the prompt.
* **Q1 shows `p02 > o00_base`** → the measured ×50 was not at the ceiling and even more delta
  pull helps; the next prompt revision should push harder on the progress term.
* **Q1 shows `p02 < o00_base`** → more pull costs what the lead-time penalty must recover, which
  is the quantitative version of the user's original diagnosis, obtained without any overshoot
  cliff.

Standing caveat: oracle-authored ablation on a known-working reward; it identifies which
guidance *form* works, not whether a search would find it.

---

## OUTCOME (fresh seeds 35000–35059, 60 episodes — the same block as the matched baseline)

| arm | `A` (pre-training) | fresh-60 | dock | mean return |
|---|---:|---:|---:|---:|
| **`o00_base` (matched baseline)** | +0.141 | **16/60 = 26.7 %** | **0.90** | 88.65 |
| `p02_progress_x200` (delta ×4 stronger) | +1.281 | **0/60** | 0.72 | −12.44 |
| `p01_proximity` (state form) | +1.162 | **0/60** | 0.02 | 4.17 |
| `p03_x200_gentle` | +1.278 | **0/60** | 0.53 | 5.82 |
| `p04_proximity_gentle` | +1.159 | **0/60** | 0.10 | 1.18 |
| **`p05_funnel`** (near × slow) | +0.517 | **0/60** | **0.07** | 3.84 |

| # | prediction | outcome |
|---|---|---|
| **Q1** | `p02` two-sided vs baseline | **worse** (0/60, and the return goes *negative*): four times the pull that works destroys delivery even though the crate still arrives (dock 0.72) |
| **Q2** | `p01 ≤ o00_base`, state form farms | **HOLDS** — 0/60 with `dock_entered` collapsing to **0.02**: the policy collects the proximity term without ever entering |
| **Q3** | `p05 ≥ o00_base` (the funnel is the strongest form of the idea) | **FAILS** — 0/60, and the dock rate falls to **0.07**: "near **and** slow" is farmable *outside* the dock |
| **Q4** | if `p02 < o00_base`, then `p03 > p02` | **not supported** — `p03` is also 0/60 (dock 0.53) |
| **Q5** | in arms that fail with `dock_entered > 0`, the dense term dominates | **HOLDS**: `p02` arrives but cannot settle (dock 0.72, negative return); `p01`/`p04`/`p05` hover instead of entering (dock 0.02–0.10) |

### The result that matters more than any single arm

**Every one of the five single-axis "guidance" edits destroyed delivery (0/60) against a matched
baseline of 16/60 in the same block.** Together with the overshoot ablation, that is **nine
perturbations of a working reward, all at 0/60**:

| perturbation | result |
|---|---|
| + overshoot cliff (o01), + both (o03), ×10 cliff (o04) | 0/60 |
| + gentleness (o02) | 1/60 |
| delta pull ×4 (p02) | 0/60 |
| + proximity (p01), + proximity & gentleness (p04) | 0/60 |
| + funnel near×slow (p05) | 0/60 |
| earlier: + guidance (r01), + guidance & gentleness (r02), ×3 guidance (r03), + stream only (r06), + stream & guidance (r07), + gentleness (r08), + stream & gentleness (r10), own speed penalty ×100 (r11) | 0/60 |
| **the recipe itself (r09 = `o00_base`)** | **23/60 and 16/60 on two independent blocks** |

So the working point is not a broad basin but a **narrow ridge**: a specific coefficient (~×50 on
the candidate's own delta-progress term) together with a specific payoff form (a dense per-step
payoff on the *instantaneous success predicate*, not on proximity and not on a positional cliff).
Perturbing either axis — up *or* down, adding or removing — falls off it. Two of the failure modes
are directly visible in `dock_entered`:

* **too much pull** (×200): arrives but cannot settle (dock 0.72, 0/60);
* **dense near/reward-shaped terms** (proximity, funnel): hovers without entering (dock 0.02–0.10).

### Honest limits of this conclusion

* Each arm is **one seed**; a 0/60 arm is bounded at ≤5 % (rule of three), so "0/60" here means
  "badly degraded", not necessarily "exactly zero".
* The five variants are **single-point samples** in a parameter space. `p02` (×200) failing while
  ×50 works shows the axis is steep, but I have not mapped the ridge: ×75, ×100 and a funnel with
  a different decay constant are untested. The load-bearing claim is therefore "the basin is
  narrow and one-axis edits fall out of it", not "the optimum is exactly ×50".
* Mechanism hypothesis for why dense additions hurt, **not yet tested**: the harness normalises
  the reward and VecNormalize clips normalised rewards, so adding a high-duty-cycle dense term
  changes how many steps sit at the clip and therefore distorts the effective per-step weighting
  of *all* terms. That would explain why adding a *positive* term for the target behaviour (p05)
  can still break delivery.

## Consequence for the prompt (this is the answer to "how do we guide EUREKA/CREATE to it")

"Reward approaching" is **not** a usable principle — all four approach forms fail. The only
measured working form is the narrow recipe, so a prompt revision (v8) has to specify it
**quantitatively**: the coefficient scale on the candidate's own delta-progress term, and the
dense per-step payoff on the *instantaneous success predicate* — plus the mechanical gate
(A > 0 and the 2×2 ordering). A directional instruction ("reward getting close", "penalise
overshoot", "add gentleness") is measured to be worse than useless on a working reward.

The cleaner methodological consequence: if the form must be spelled out this tightly, the search's
role collapses to **parameter tuning inside a specified form** — which is measurable, and which is
exactly what the operator ablation should test next (seed the loop off-ridge and see whether its
diagnosis recovers a known-good parameter).
