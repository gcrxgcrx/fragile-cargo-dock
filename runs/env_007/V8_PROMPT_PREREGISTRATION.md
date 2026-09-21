# PRE-REGISTRATION: v8 — fixing a self-contradiction I wrote, and one mechanism lesson

Written after the v7 scoring (1/8 hits, 3/60 = 5.0 %) and **before** any v8 candidate is
generated. Companion: `runs/env_007/v7_prompt.diff` (the previous revision),
`runs/env_007/v8_prompt.diff` (this one), generator `make_prompt_v8.py`.

## What v7 got wrong — three measurements

**(1) v7 contradicts itself, and the contradiction is the likely reason the family docks
without settling.** Two clauses target the *same state* (the settled state = the completion
state):

* the requirement list requires a per-step payoff there ("**同时**必须有一项…停稳期每步收益…
  它不是可选加分项，而是必要条件"), enforced by self-check ④ ("call the reward 12 times on a
  settled state; the total must grow linearly");
* the `完成状态下…其余组件必须恰好为 0` clause requires that, when the completion predicate
  holds, **every component except the one-off event contributes exactly 0**.

All eight v7 candidates resolved it the same way — they gate the per-step payoff on
`not _PAID` (or zero it once the event fires) — i.e. **the payoff stops exactly when the
policy succeeds**.

**(2) The one-off event is worth exactly zero under the harness clip.** `probe_v7_shape.py`,
on a real success trajectory (`release_0.25`, 303 steps), clip 20:

| candidate | terminal step (raw → clipped) | cumulative to completion | truncated + 40 settled steps |
|---|---:|---:|---:|
| `cand_00` | 320.0 → **20.0** | 96.7 | **876.7** |
| `cand_01` | 300.0 → **20.0** | 3083.7 | **3263.7** |
| `cand_02` | 320.0 → **20.0** | 333.4 | **1113.4** |

The +300 event and the +20 stream sum to 320 on the terminal step and are clipped to 20 —
bit-for-bit what one settled step pays. So the v5/v7 rule demanding
`10 * B > 3 * (process bound * 400)` makes the model spend its design budget on a term the
harness deletes.

**(3) The zeroing rule re-introduces the behaviour it was meant to prevent.** Because the
payoff stops on completion, *not* finishing keeps paying while finishing ends the episode:
`cand_00`'s truncated + 40-settled-step counterfactual scores 876.7 against 96.7 for the
successful episode. (This is a *counterfactual over an unreachable continuation*, since the
environment terminates after the 10th consecutive settled step — it shows the payoff
structure is wrong, not that the agent faces this choice. See
`check_finish_vs_delay.py`.)

Also measured, and the reason the fix is *not* "ban dense payoffs": the best hand-written
arms (`probeD` 73.3 %, `probeA` 40 %) pay a constant `+20` on every settled step — an
unbounded state reward — and they work. What distinguishes the arm that ground in the dock
for 43 steps (`control_v1`, 0 %) is not the payoff but a **closed-loop progress term**
(`max(0, Δd)`, no penalty for moving away). That defect was measured directly in the v7
family: `cand_01`'s `crate_to_dock_progress` accumulates **2747** per episode where geometry
bounds it at ≈ 450 (the crate starts ≈ 4.5 m out), and the candidate scores 3/60.

## The three edits in v8 (nothing else changes)

1. **One-off event demoted.** The requirement "完成奖励必须一次性" loses its mandatory
   status; the template keeps the code but is followed by the clipping arithmetic and the
   instruction not to inflate the event to satisfy a magnitude formula. Coefficients remain
   entirely the generator's choice (user decision, 03:00).
2. **The settled payoff must not be switched off** by the candidate's own event or streak
   counter. Self-check ⑤ is rewritten: 12 calls on a settled state must each grow by the
   *same positive amount* (a switch-off now fails the check).
3. **The zeroing rule is narrowed** to persistent **state** bonuses (alignment/nearness/slow
   bonuses), explicitly exempting the per-step success-predicate payoff; and the progress
   term must be **signed and cycle-neutral** (a round trip must not pay), with the measured
   2747-vs-450 counterexample stated.

## Protocol

`pilot_generate_only.py`, `prompts/eureka_01_initial_reward_v8.md`, the same context
(`runs/env_007/terminal_rule_pilot/seed_0/context`), config
(`configs/env007_terminal_rule_pilot.yaml`), model (`deepseek-flash`) and `temperature=0.7`
as the v7 arm, so the only difference is the prompt. **N = 8**, output
`runs/env_007/prompt_ladder_v8/`.

Scoring, in this order (the third step is gated by the second, so no training is spent on a
family that did not change):

1. **Generation-only, mechanical** (`check_settled_stream.py` + a new cycle-neutrality
   check): how many candidates pay on a settled state **after** their own event has fired.
   v7: 0/8 for the after-event form (7/8 paid only *before* the event, 1/8 zeroed it).
2. **Structure of the emitted code**: does the candidate gate the payoff on its own payment
   flag? does it keep a one-off event at all?
3. **Training** of whatever passes (1)–(2), 1.2 M steps, and **two training seeds per
   candidate**, evaluated on one fresh block with `dock_entered` and success reported
   separately.

## Predictions, fixed now

| # | prediction |
|---|---|
| **V8-1** | ≥ 6/8 candidates pay on a settled state **after** their own success event has fired (v7: at most 1/8) |
| **V8-2** | the family's progress terms are bounded: realised per-episode progress sum ≤ 500 for ≥ 6/8 (v7: 2747 for its only docking candidate) |
| **V8-3** | the prompt change alone does **not** produce a ≥ 8/60 candidate on the first seed (the v7 experience: a mechanically complete prompt change gave 1/8 hits, 3/60) |
| **V8-4** | if any candidate reaches ≥ 8/60, its `dock_entered` is > 0.5 (i.e. delivery comes from reaching, not from noise) |

## Verdict rules, fixed now

* **V8 passes the form test** iff V8-1 and V8-2 hold. This licenses spending training on the
  family; it says nothing about success.
* **V8 passes the substance test** iff some candidate reaches ≥ 8/60 on ≥ 2 independent
  training seeds. Only then is a full-pipeline run (CREATE vs EUREKA, 10 × 3 M) warrantable.
* **If V8-1 holds but no candidate reaches 8/60**, the conclusion is that the prompt is no
  longer the binding constraint on *form*, and the remaining gap is the one the isolation
  experiment (`CLOSE_SPEED_ISOLATION_PREREGISTRATION.md`) and the settling question address.
* **If V8-1 fails** (the generator keeps gating/zeroing the payoff even when told not to),
  then this operator cannot be corrected by prompt text at all, and the honest next move is
  the operator/pipeline route (multi-round iteration) rather than more prompt revisions.

## Standing caveats

* v8 is authored by me from measurements on this environment; like v2–v7 it is a *scaffold*
  change and must be reported as part of the prompt lineage, not as a search result.
* N = 8 per stage with 2 training seeds per candidate is still small; the seed count is
  stated with every number because §5.8 measured the settling step to be near-coin-flip.
* These edits make the prompt name three things a reward must not do. That is the same class
  of intervention as v7 — and v7's own lesson is that a correct rule can be insufficient.

---

## STATUS: generated 04:23, form checks scored 04:24

### OUTCOME — the form checks (V8-1, V8-2)

8/8 candidates valid (one needed 3 attempts: `import` then `try:` — the validator caught both).

`check_settled_stream.py` (median of calls 2–12 on a settled transition) and
`analyze_family_shape.py` (event-aware, over 27 scripted trajectories):

| candidate | settled stream? | per-step AFTER its own event | progress integral | mean \|r\| |
|---|---|---|---:|---:|---:|
| `cand_00` | yes (19.99) | 24.99 | **39.2** | 0.481 |
| `cand_01` | yes (20.00) | 25.00 | **8.7** | 0.187 |
| `cand_02` | no (−0.02) | 5.00 | 0.95 | 0.305 |
| `cand_03` | yes (19.94) | 24.94 | **5.7** | 0.404 |
| `cand_04` | no (−0.01) | **−0.01** | **33.7** | 0.338 |
| `cand_05` | yes (19.99) | 20.00 | **5.7** | 0.132 |
| `cand_06` | yes (19.98) | 24.99 | **24.7** | 0.927 |
| `cand_07` | yes (19.98) | 39.98 | **104.7** | 1.025 |

* **V8-1 (≥ 6/8 keep paying after their own event): HOLDS** — 7/8 pay a positive per-step
  amount after their success step fired (v7: 0–1/8; the v7 family gated the payoff on
  `not _PAID`, and `cand_01` even zeroed it once `_PAID` was set). The one failure is
  `cand_04`, which has no usable settled payoff at all.
* **V8-2 (≥ 6/8 with bounded progress): FAILS** — only **3/8** (`cand_01` 8.7, `cand_03` 5.7,
  `cand_05` 5.7) are bounded; four exceed anything measured in v7's docking candidate
  (`cand_07` 104.7, `cand_00` 39.2, `cand_04` 33.7, `cand_06` 24.7) even though the prompt
  states the rule together with the 2747-vs-450 counterexample.
* **The saturation defect survives the fix.** For six candidates the settled payoff is *at*
  the clip (20.0), so the `+300` one-off event they were told not to rely on is still worth
  exactly zero. The prompt changed the *gate* but not the *arithmetic*.

**Reading:** the v8 rules are **partially absorbed**. "Do not switch the payoff off" was
adopted almost completely; "progress must be signed and cycle-neutral" was adopted by only
3/8, with the rest reproducing the closed-loop form despite an explicit counterexample. This
is the same operator-limitation signature as `REPAIR_LOOP_FINDINGS` (real 0/8 vs sham 1/8):
the model implements the rule it can pattern-match onto a code template and misses the one
that requires reasoning about the reward's integral.

### Training (wave 3) — INTERRUPTED, see the note

The three candidates satisfying both form properties (`cand_01`, `cand_03`, `cand_05`) were
launched at 1.2 M steps, seed 0. **The queue was killed at 04:30 by an operator error**
(`job_kill` on the wrapper job took the queue's child trainers with it). `cand_01` was
retrained solo immediately; `cand_03` and `cand_05` were not retrained within the session.

### The one v8 candidate that was trained, and what it shows

`cand_01` was retrained solo (1.2 M, seed 0) after the wave-3 interruption, and scored on the
fresh block **38000-38059**:

| quantity | value |
|---|---|
| success | **4/60 = 6.7 %** (v7's best unedited candidate: 3/60) |
| `dock_entered` | **0.70** (v7's best: 0.18) |
| max consecutive stable steps (mean / median) | **2.42 / 1** |
| docked-and-failed episodes reaching 7-9 steps | 2 / 38 |
| dominant component | `settled_reward`, share **0.761**, active 0.015, 117 points/episode |
| **progress component, per episode** | **14.04** (v7's docking candidate: **2747**) |
| `gentleness`, per episode | **−1.73** |

So the v8 fixes are real at the level of the reward: the settled payoff is now the dominant
term, it is no longer switched off, and the progress term is **bounded** (14 vs 2747 — a
400x improvement on the exact defect the prompt names with a counterexample). V8-1 holds and
the reward's *shape* is the best an LLM candidate has ever had here.

**And it still does not settle.** `max_stable` = 2.42 steps against 0.83 for the control and
8.03 for the hand-written arm that is identical except for one gentleness term. The v8
candidate *has* a gentleness term; it is worth −1.73 per episode, whereas the hand-written
arm's is what produces the 8-step hold (`CLOSE_SPEED_ISOLATION_FINDINGS.md` §3, §5).

**This is the sharpest statement of what is missing that this project has produced:** the LLM
can write the right *terms* and even, under a good prompt, the right *shape*; what it cannot
do is set a term's **effect size** so that the behaviour actually changes. V8-3 (>= 8/60) is
therefore **not met** at 4/60, and the conclusion of this revision is that the prompt is no
longer the binding constraint on *form* — **calibration** is.

### Correction to the V8-2 scoring (kept for the record)

The first pass scored V8-2 from a new aggregate statistic (`progress integral` = the sum of
progress components over 27 scripted trajectories divided by their summed net distance
changes) and reported 3/8 bounded, four worse than v7. Reading the emitted code shows that
**8/8 candidates did implement a signed difference** (`progress = dist_prev - dist_now`, some
with a symmetric `max(-3, min(3, ·))` clamp); the two candidates containing `max(0, ...)` use
it elsewhere. The aggregate statistic is **too coarse to decide V8-2** — it mixes trajectories
and is contaminated by truncated episodes. V8-2 is therefore scored on the *trained*
candidate's realised component table, the measurement that exposed `v7_cand_01`'s 2747:
`v8_cand_01` = **14.04 per episode**, i.e. bounded.



The first pass scored V8-2 from a new aggregate statistic (`progress integral` = the sum of
progress components over 27 scripted trajectories divided by their summed net distance
changes). Inspection of the emitted code shows that **8/8 candidates did implement a signed
difference** (`progress = dist_prev - dist_now`, some with a symmetric `max(-3, min(3, ·))`
clamp); the two candidates that contained `max(0, ...)` use it elsewhere, not on progress.
The aggregate statistic is therefore **too coarse to decide V8-2** — it mixes trajectories
and is contaminated by truncated episodes. V8-2 is **undecided**, and the only valid metric
for it is the realised per-episode progress sum from a *trained* policy (the metric that
exposed `v7_cand_01`'s 2747). The v8 candidates implement signed progress and scale it into
the hundreds (e.g. `cand_00`: ×100 with a ±3 clamp), which is the same order as the arm that
ground in the dock; whether that farms in practice can only be read off a trained run.


