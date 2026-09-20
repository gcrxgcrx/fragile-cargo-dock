# Ridge-width bisection: design and outcome

> **HONESTY BANNER — this one was NOT pre-registered.** Unlike every other experiment in this
> session, I built the two variants (`make_ridge_variants.py`), launched the queue and only then
> wrote this file. The design was recorded in the variant headers and in the session record
> before the runs, and the two bracketing points (coefficient 600 works, 2400 fails) were already
> measured — but there is **no fixed prediction on disk for the middle points**, so the result
> below is descriptive, not a passed/failed prediction. This is recorded rather than quietly
> omitted.

## Design

Base: `runs/env_007/repair_test/r09_progress_x50_stream/reward_v1.py` — the two-edit repair of
`L2/cand_00`, whose delta-progress coefficient is 600 (i.e. ×50 of the original 12). Single-axis
change: the coefficient only. Same 1.2M protocol, seed 0, scored on fresh seeds **35000–35059**,
the same block as the matched baseline `o00_base` (a byte-identical copy of the base reward).

Bracketing points already measured in that block: coefficient 600 → **16/60**, coefficient 2400
(arm `p02`) → **0/60**.

## Outcome

| coefficient | ×original | fresh-60 | `dock_entered` | mean return | training-time terminations |
|---|---:|---:|---:|---:|---:|
| 600 (baseline) | ×50 | 16/60 = 26.7 % | 0.90 | 88.65 | 9/20 |
| **900** (`q01_progress_x75`) | ×75 | **0/60** | **0.07** | 4.40 | **0/20** |
| **1200** (`q02_progress_x100`) | ×100 | **31/60 = 51.7 %** | **0.95** | **164.03** | 9/20 |
| 2400 (`p02_progress_x200`) | ×200 | 0/60 | 0.72 | −12.44 | — |

`q02` at **31/60 = 51.7 %** is the **best result of the entire session** (against hand-written
`probeA` 43.3 % / `probeC` 98.3 %, and the earlier best LLM-derived candidate 5/60). One-sided
Fisher for 31/60 vs 0/60 is astronomically small; the meaningful comparison is against the
baseline in the same block: **31/60 vs 16/60**, Fisher one-sided p ≈ 0.004.

The mechanism readouts agree that the arms trained to *different* policies, not that the eval
merely wobbled: `q01` (×75) shows **0/20 terminations**, `success_event` active rate **0.0000**
and `dock_entered` 0.07 (hover, never enters), exactly the failing signature; `q02` (×100) shows
9/20 terminations and `success_event` active rate 0.0012 with a mass split of
`REPAIR_settled_stream` 40 % / `crate_to_dock_progress` 37 % / `success_event` 12 % — the same
signature as the working baseline.

## What this says

**The dependence on the reward's scale is rugged, not a smooth band.** Adjacent points differ
qualitatively: ×50 works, ×75 collapses, ×100 works *better than either*, ×200 collapses. So the
earlier phrasing "the working point is a narrow ridge" is too kind — a ridge is at least locally
smooth. The correct description is a **jagged success surface in a single scalar**, which is a
stronger version of the theoretical difficulty in `HANDOFF_QUESTIONS.md` P1–P5: the *form* can be
specified, and even then the *coefficient* is a lottery.

## The instrument cannot see it (measured, same run)

`advantage_scale_probe.py` on the four reward files:

| reward | coefficient | `A` (per step) | actual fresh-60 |
|---|---:|---:|---:|
| `o00_x50` | 600 | +0.141 | 16/60 |
| `q01_x75` | 900 | +0.331 | **0/60** |
| `q02_x100` | 1200 | +0.521 | 31/60 |
| `p02_x200` | 2400 | **+1.281** | **0/60** |

`A` is **smoothly monotone in the coefficient** while the outcome is jagged — it ranks the worst
arm (×200) highest. So:

* `A` is a **necessary-condition** check only. A mechanical gate built on `A > 0` (the v8 plan)
  would **pass ×75 and ×200, both of which score 0/60**. That plan is therefore insufficient as
  designed and must not be presented as a fix;
* this is a **second independent negative** for the advantage probe as a selector (the first:
  the `L0` family scores highest on `A` while never succeeding).

## Caveat, and the measurement now running

One seed per point. A 0/60 arm is bounded at ≤5 % (rule of three), and the same reward file has
already produced 23/60 and 16/60 on two different blocks — so run-to-run variation exists. A
swing from 0 to 31 is ~4× that observed variation *and* the two arms show different in-training
termination statistics, which argues for a real difference — but a single seed per point cannot
settle it.

**Seed-robustness runs are in flight** (`runs/env_007/ridge_seeds_spec.json`, seeds 1 and 2 for
both ×75 and ×100, same block):

* if ×75 stays ≈0 across three seeds and ×100 stays high → **the surface is genuinely rugged**,
  local/gradient calibration is misled, and the viable strategy is sample-and-train;
* if ×75's seeds spread widely → the surface is a **noisy success probability** rather than
  rugged, and every n=1 comparison in this session (including the ablation drops) needs that
  caveat attached.

Either outcome changes the plan, so it is the right next measurement and it is already running.
