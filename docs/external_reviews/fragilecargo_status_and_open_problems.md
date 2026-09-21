# HANDOFF — FragileCargoDock-v0 reward-search study: status and open problems

**Purpose of this document.** A complete, self-contained statement of where this project
stands, what has been *measured* (as opposed to argued), and which problems remain open —
written so that an outside reviewer can reason about it without access to the repository.
Numbers in **bold** are measurements with raw artifacts on disk; every claim is tagged with
the experiment that produced it.

**One-line summary.** Hand-written observation-only rewards for this task reach 40–77 %;
every reward produced by an LLM search scores 0–5 %; an *oracle* two-edit repair of an
LLM reward reaches 38 %; and we can now measure the defect (the reward's decisive term is
never active under the policy's own behaviour) but **no training-free statistic we have
tried predicts which reward will learn** — five independent instruments have been tested
and rejected.

---

## 1. The task

`FragileCargoDock-v0`: top-down Box2D, zero gravity, one cart and one freely-moving
"fragile" crate. A partition wall with a narrow gap separates the start area from a marked
docking bay. **The cart has no brake and no gripper**: it can only push, and the crate
decelerates only through floor damping. So the only way to satisfy the low-speed
requirement is to *release early* and let drag carry the crate in — a precision–timing
coupling that is the task's core difficulty.

* Observation: `Box(19,)`, all entries clipped to `[-2, 2]`. Cart pose/velocity,
  crate-relative position in the body frame, crate world velocity, crate heading,
  crate-to-dock signed offset (`obs[12], obs[13]`), a binary contact flag (`obs[14]`),
  three short-range static-obstacle sensors, and an episode time fraction (`obs[18]`).
* Action: `Box(2,)` in `[-1, 1]` — drive force (≤ 34 N) along the heading, steering torque.
* Episode: 400 steps (dt = 1/30 s).
* **Success** (terminates the episode): crate fully inside the dock rectangle
  (`|obs[12]| ≤ 0.024`, `|obs[13]| ≤ 0.030`), heading error < 30°, crate speed < 0.05 m/s,
  **held for 10 consecutive steps**.
* **Failure** (terminates): crate or cart leaves the field, or 3 hard collisions
  (per-step peak cart↔crate contact impulse > 5 N·s).
* Timeout at 400 steps is `truncated`, not success.

The environment's own ("native") reward, hidden from the generating LLM, is:

| term | weight | form |
|---|---|---|
| `approach_cargo` | +1.0 / m | cart→crate distance closed (potential-difference shaping) |
| `progress` | +1.0 / m | crate→dock distance closed (potential-difference shaping) |
| `dock_enter` | +5.0 | once, first step fully inside |
| `roughness` | −0.02 / (N·s) | proportional to contact impulse |
| `action_cost` | −0.0005 | squared action |
| `time_cost` | −0.002 / step | |
| `hard_hit` | −0.5 | per hard impact |
| `success` / `failure` | +300 / −100 | terminal |

Native return ≈ `10 + 300 × delivery_rate`.

---

## 2. Baselines and the hand-written ceiling

PPO, `n_envs = 6`, γ = 0.999, `normalize_reward = true`, `ent_coef = 0.005`. Evaluation is
always on **held-out seed blocks** (60 fresh episodes unless stated).

| policy | budget | fresh success |
|---|---:|---:|
| random | — | 0 % |
| scripted heuristic pusher | — | 50 % |
| **PPO on the native reward, bypassing the wrapper** | **3.0 M** | **96.8 %** |
| **PPO on the native reward, through the wrapper, clip 20** | **1.2 M** | **90.0 %** |
| native reward, through the wrapper, clip 600 | 1.2 M | 80.0 % |

`normalize_reward` is the decisive training knob for the native reward: γ = 0.999 alone
gives **0 %**, γ = 0.999 + normalisation gives **80 %** (1.2 M, 5 envs). The native reward
itself had to be hand-debugged: its original per-step contact bonus created a trapping
local optimum (the cart leaned on the jammed crate for ≈ +5/episode); replacing it with an
impulse-proportional cost destroyed exploration; adding the `approach_cargo` potential term
restored it (0 % → 30–100 % across seeds at 750 k steps).

**Hand-written reward arms** (the reward *must* be a Python function of
`(obs, action, next_obs, info, training_progress)`, with `info`'s diagnostic fields
forbidden; the per-step reward is clipped to `[-20, 20]` by the harness):

| arm | what it is | fresh-60 | Fisher vs 0/60 |
|---|---|---:|---|
| `control_v1` | incremental crate progress + cart approach + +20/step settled stream + boundary guard | **0 %** | — |
| `probeA` | `control_v1` + a true contact-impulse penalty (reads `info`; diagnostic only) | 40.0 % | <1e-4 |
| `probeB` | per-step stream replaced by the native one-off terminal event (reads `info`) | 45.0 % | <1e-4 |
| `probeD` | `+` **observation-only closing-speed penalty** ("gentleness") | **73.3 %** | <1e-4 |
| `probeE` | obs gentleness + a module-state one-off success event via `obs[18]`, no `info` | **65.0 %** | <1e-4 |
| `probeC` | true impulse + event form | **98.3 %** | <1e-4 |

So the observation-only contract **can** express a solvable reward, and the harness is not
the bottleneck: the native reward still scores 90 % *through* the wrapper at clip 20.

**The LLM search has never produced a usable reward.** Best ever before the work below:
**5/60 = 8.3 %** (a single pilot candidate, which then collapsed to 0/60 at the full 3 M
budget). CREATE and EUREKA-style population search at 10 runs × 3 M each both landed at
**0–1/60**.

---

## 3. What has been established (all measured, all negative)

### 3.1 Structural and behavioural checks carry no information about learning

* **Structural probes** (term presence and dominance ratios on hand-built states): eight
  candidates chosen to span the checks from "all four pass" to "none pass" were trained at
  1.2 M — **all eight scored 0/60**. With earlier ones, that is 11 scaffold-compliant LLM
  rewards with zero deliveries.
* **Trajectory-ranking check** (does the reward order *physically reachable* trajectories
  the way the task does?): `control_v1` scores **0 %** and yet ranks success above every
  failure with accuracy **1.000** and separation 199.6 — exactly like the 73.3 % probe.
  So **ranking correctness ≠ learnability**; no training-free ordering check can select.
* **Short-training rungs as cheap proxies**: at 0.6 M steps the rank correlation with
  1.2 M success is **ρ = −0.486** (it ranks the 0 %-at-1.2M control *second of six*) and
  seed noise (sd 0.236) exceeds between-candidate spread (0.149). At 1.0 M steps
  ρ = **+0.371**, still below the 0.60 bar, and the readout certifies the 0 % control in
  3/3 seeds. **Selection cannot be made cheaper than evaluation on this environment.**
* **Per-step advantage probe** `A` = mean per-step generated return over seven scripted
  heuristic controllers minus that of `idle`, computed with **no training**. It answered a
  real question — `A ≤ 0` means *inaction is optimal*, which is exactly the whole immobile
  family — and it predicted, before their results were read, which repair arms would move
  the cart and that scaling a candidate's own speed penalty ×100 would be catastrophic
  (`A` = −12.8). **But it is rejected as a selector**: AUPRC 0.413 against prevalence 0.357,
  and the three *failing* L0 candidates hold the largest `A` of the pool (+0.39, +0.32,
  +0.17), above every working arm — a dense farmable term pays more while acting than while
  idling.
* **Four further statistics tested in the most recent round** (shaping:terminal ratio,
  "hover wage", dock-dwell reward, dock-pay share) over 23 arms with known outcomes:
  Spearman ρ against success = **0.275, 0.042, 0.204, −0.112** respectively (bar was 0.60).
  Same corpus, `A`: ρ = 0.292. **None is a selector, and none is even a useful correlate.**
  A striking internal control: `control_v1` and `probeD` have *identical* values on all
  four statistics and score **0 % vs 73.3 %** — the difference between them is one extra
  term (a closing-speed penalty). Conversely, arms with the *same* shaping:terminal ratio
  score 0, 24 and 44.

### 3.2 The prompt scaffold I wrote was itself a leak

Dose–response over 16 candidates per level, generation only, identical context/model/
temperature:

| check | L0 (interface only) | L1 (the paper's prompt) | L2 (my scaffold) |
|---|---:|---:|---:|
| writes a one-off terminal event | 1/16 | 0/16 | **16/16** |
| writes a gentleness term | 0/16 | 2/16 | **15/16** |
| all four structural checks pass | 0/16 | 0/16 | **11/16** |

L0 ≈ L1, so the "leak" is specifically the rules I added on top of the paper's prompt.

### 3.3 Why the LLM rewards fail — mechanism

Every arm that works puts ≈ **96 %** of its reward mass in a positive, sparse term **that is
actually reached** (a settled-state stream, or a one-off completion event). Every failing
LLM arm puts **0 %** there: its mass sits in a penalty, in a **farmable dense term**, or
nowhere at all (one candidate's reward is identically 0 on the trajectory its own policy
produces, so there is no gradient). Concretely measured on the single v7 candidate that
ever docked (`v7_cand_01`, A = +4.28, 3/60):

* its `crate_to_dock_progress` term accounts for **98 %** of the reward, active in **45 %**
  of steps, with a realised per-episode sum of **2747** — while the geometry allows at most
  ≈ 450 (the crate starts ≈ 4.5 m from the dock; a true incremental term can only be paid
  for the net distance closed). **It is collecting reward over a closed loop.**
* its settled-state payoff component has `magnitude_share` **0.000** and `active_rate`
  **0.000** at the end of training.
* for comparison, the working hand-written arms' progress components have per-episode sums
  of **5** (`probeD`, `probeE`, `probeA`), **229** (`r09`) and **401** (`q02`).

### 3.4 The evidence channel is not the bottleneck

The pipeline's own reflection report — verified to contain the decisive component-activity
evidence (`success_event` activation 0 % for the failing candidates, 100 %-active dense
terms for the control) — was given to the repair operator, with a **sham control** in which
every per-component table's rows and each value column were independently shuffled:
**real 0/8 repairs vs sham 1/8**, Fisher p = 1.0000. Prediction recorded in advance: adding
evidence channels (including the planned trajectory-evidence extension) is worthless on
this environment until iteration or a different prompt/operator shape is tested.

---

## 4. The one positive result: an oracle two-edit repair

An *oracle-authored* study (its purpose is to measure the size of the target, not its
discoverability) applied two localized edits to a 0/60 LLM candidate and got the first
LLM-derived reward that ever delivered the crate:

| arm | edit | fresh-60 | dock rate | mean native return |
|---|---|---:|---:|---:|
| base candidate | — | 0/60 | 0.00 | −1.20 |
| byte-identical copy | none | 0/60 | 0.00 | −1.20 |
| `r04` | candidate's own approach coefficient 12 → 600 (×50) | 0/60 | **0.97** | +9.14 |
| `r06` | **only** a dense +20/step payoff on the success predicate | 0/60 | 0.00 | −1.20 |
| **`r09`** | **both of the above** | **23/60 = 38.3 %** | **0.98** | +124.27 |
| `r08` | ×50 + an added gentleness penalty | 0/60 | 0.00 | +1.67 |
| `r11` | ×50 + the candidate's own speed penalty ×100 | 0/60 | 0.00 | −1.65 |

* The two halves are **individually useless and jointly decisive**: scale alone docks but
  never settles; the dense payoff alone is *bit-for-bit equivalent to the copy control*
  because its predicate is never reached.
* The recipe **replicates** on a second, independently generated candidate: 23/60 again.
* Adding gentleness **hurts** (0/60, dock 0.00); raising the candidate's own speed penalty
  ×100 is worse than the base (predicted in advance by the advantage probe).
* Mechanism confirmed: `success_event` activation 0.0000 → 0.0012/0.0017, contributing
  12–29 % of realised mass.

**A separate repair attempt shows the limit of "fixing the stated bug":** one candidate's
alignment term was *inverted* (`1 − |obs[10]|` while the dock angle is 0, so it rewarded
holding the crate perpendicular — a state that can never succeed). The one-subscript fix
moves the training-free ordering check from 0.038 to **0.702** and the arm **still trains to
0/60**, because its per-step advantage stays at `A ≈ −0.0004`.

---

## 5. The prompt revision (v7) and what it achieved

The prompt in use forbade the very ingredient the repair needed: "the completion reward must
be a one-off event; do **not** pay per step on the success state (violation = invalid)",
enforced by a self-check that told the model to rewrite any such payoff. The ban's
justification was a real measurement whose *attribution* was wrong: the arm that ground in
the dock for 43 steps did so because its approach term was far too weak (`A = −0.26`), not
because of the per-step payoff.

**v7** replaced the ban with a conditional rule requiring *both* a one-off completion event
and a per-step settled payoff, and inverted the self-check.

| v7 outcome | value |
|---|---|
| candidates that pay per step on a settled state | **8/8** (previous family: 0/2) |
| candidates with `A > 0` | 4/8 (previous family: 1/8) |
| first unedited LLM candidate ever to enter the dock | `v7_cand_01`, `dock_entered` **0.18** (all 35+ earlier ones: 0.00) |
| honest rate | **1/8 hits, 3/60 = 5.0 %**, one-sided Fisher p = 0.333 vs the 0/16 baseline (inside the noise floor) |

**Post-hoc analysis of v7 (new, and not yet acted on).** v7 contains a **flat
self-contradiction**: one clause requires a per-step settled payoff that must grow every
step (self-check: calling the reward 12× on a settled state must increase linearly), while
another clause requires that "when the completion predicate holds, **every component except
the one-off event must contribute exactly 0**". The two clauses target the *same state*. All
eight candidates resolved it the same way — by gating the per-step payoff on "the one-off
event has not fired yet" or by zeroing it once it fires. Measured consequence on a real
success trajectory under the harness clip of 20:

* the one-off event's +300 is clipped to the same value as a single settled step, so its
  contribution at the terminal step is **exactly zero**;
* and because the settled payoff switches off, **not completing is worth about +780 more
  than completing** for one candidate (876.7 vs 96.7 cumulative, clipped) — the reward
  literally pays better for hovering in the dock than for finishing.

This is a defect *I wrote into the prompt*, and it is the clearest single explanation for a
family that docks but cannot settle. It is not yet fixed.

---

## 6. What is settled, and what is genuinely hard

### 6.1 The two-factor structure of the difficulty

`dock_entered` (does the policy get the crate to the dock) is set fairly reliably by the
reward's *scale*; **whether the 10-step hold ever completes is near-coin-flip at fixed
settings**. Single-axis sweep of the one coefficient the working recipe fixes (same fresh
block, 1.2 M protocol, then seeds 1–2 added):

| coefficient | seed 0 | seed 1 | seed 2 | mean | `dock_entered` per seed |
|---|---:|---:|---:|---:|---|
| ×50 (600) | 16/60 | — | — | (16/60) | 0.90 |
| ×75 (900) | 0/60 | 4/60 | 0/60 | **2.2 %** | 0.07 / 0.48 / 0.00 |
| ×100 (1200) | 31/60 | **43/60** | 0/60 | **41 %** | 0.95 / 0.78 / 0.95 |

The decisive datum is ×100 seed 2: it docks in **95 %** of episodes and scores **0/60**.
So `success ≈ P(reach) × P(settle)`, with the second factor near-coin-flip at fixed
settings. **Consequences:** single-seed candidate comparisons are one lottery draw; success
and `dock_entered` must be reported separately and with the seed count; and the advantage
probe is blind to this (it is smoothly monotone in the coefficient — +0.141 / +0.331 /
+0.521 / +1.281 — while the outcomes are jagged, so it ranks the worst arm, ×200 = 0/60,
highest).

User decision already recorded: **no further multi-seed confirmation runs**; the coefficient
is deliberately left to the LLM/search, with the caveat that scale is a first-order,
non-monotone variable here, so a *sampling* loop suits it better than a gradient-following
one.

### 6.2 Nine single-axis "expert" edits destroy a working reward

Against a matched **16/60, dock 0.90** baseline in the same seed block:

| the idea | fresh-60 | dock |
|---|---:|---:|
| a positional overshoot cliff (−20/step, and −200/step) | **0/60** (both) | 0.00 |
| an extra gentleness / closing-speed penalty | **1/60** | 0.35 |
| a proximity (state) reward for approaching | **0/60** | 0.02 |
| the same idea in delta form, ×4 stronger | **0/60** | 0.72 |
| a "near AND slow" funnel | **0/60** | 0.07 |

Mechanism, measured: with no brake the only route to the dock is to push and let drag stop
the crate, so arriving trajectories necessarily pass near the far edge; a positional cliff
at −20/step makes the **approach itself** unprofitable, `A` flips from +0.141 to −0.253, and
the optimum becomes *not approaching*. Every direction a domain expert reaches for first
fails here.

### 6.3 The genuinely unexplained observations

These are the facts that our current theory does **not** account for, and they are where an
outside perspective is most likely to help:

1. **`control_v1` (0 %) and `probeD` (73.3 %)** differ by one term — an observation-only
   closing-speed penalty — yet are *identical* on every training-free statistic we have
   computed (shaping:terminal ratio 9.22, hover wage 3.05, dock-dwell 1.26, dock share
   0.063). We have not run the single-variable isolation of that term against a matched
   baseline.
2. **Settling is near-coin-flip at fixed settings** (×100: dock 0.95 with 0/60 in one seed,
   43/60 in another). Nothing we have predicts which seed settles. Is this genuine
   optimisation variance, an artefact of reward normalisation, or something about the
   environment's chaotic contact dynamics?
3. **A dense settled payoff helps hand-written arms and *hurts* LLM candidates** (stacking
   gentleness on a candidate that already penalises closing speed destroys locomotion:
   0/60, dock 0.00). Why does the same ingredient transfer so badly?
4. **The clip arithmetic.** Under `reward_clip = 20`, a one-off +300 terminal bonus is worth
   exactly one stream step. Yet the native reward — whose terminal +300 is also clipped to
   +20 through the wrapper — still reaches 90 %. So the completion signal survives clipping
   when it is the only positive sparse term, and is destroyed when a much larger dense term
   competes with it. Is that the whole story?
5. **Reward normalisation may invert the intended weighting.** With `normalize_reward =
   true`, a reward whose per-step magnitude is large has its terminal spike divided by a
   large running standard deviation, whereas the near-sparse native reward has its terminal
   spike *amplified*. This has never been tested directly (e.g. native vs a dense arm at
   `normalize_reward=false`), and it could explain why "more dense shaping" hurts here.

---

## 7. Constraints any proposed fix must respect

* **Generation model is fixed**: `deepseek-flash` only (a stronger generator's API is not
  available — that hypothesis is closed by the user), `DEEPSEEK_THINKING=disabled`.
* **The reward function may only use the observation and action**; the diagnostic `info`
  fields (impulse, distance, stable-step counter, termination reason) are forbidden. This is
  the contract the study is about.
* Per-step reward is clipped to `[-20, 20]`; reward normalisation is on.
* **Budget reality**: a 1.2 M-step training is ≈ 3 min wall-clock at 4-way parallelism; the
  full pipeline (10 runs × 3 M) is ≈ 1.5 h for the population method and ≈ 5 h for the
  single-lineage method. Small, well-chosen experiments are cheap; a full pipeline run is
  not, and it only measures *method comparison*, not mechanism.
* **Evaluation**: 60-episode held-out seed blocks. Seed blocks are consumed by use
  (30000–30059, 32000–32059, 33000–33059, 34000–34059, 35000–35059, 36000–36059 used so
  far), so a proposed design should say which block it will consume.
* The project's own rule (learned the hard way): **any new selector must be pre-registered
  with a bar, and must be tested against arms whose outcome is already known** — five
  training-free instruments have now been rejected that way.

---

## 8. The open problem, stated plainly

**We can specify a working reward, and we can measure why the LLM's rewards fail, but we
cannot predict which reward will learn, and we do not know why the gap between a 40–77 %
hand-written arm and a 0–5 % LLM arm is so large given that the observation-only contract
can express the former.**

The three sub-questions we would most like an outside opinion on:

* **(Q1) What is the missing ingredient?** Our best current candidate explanations are:
  (a) the decisive term must be *active under the policy's own trajectory* — an
  exploration/reachability problem, not a shaping problem; (b) the reward must be
  *impossible to farm* (bounded, potential-difference shaping) so that the only way to increase
  return is to finish; (c) the reward's terminal/rare positive term must survive the
  interaction of the per-step clip and reward normalisation. Which of these is testable
  most cheaply, and is there a fourth we have not considered?
* **(Q2) Is `P(learn) ` even predictable from the reward function alone on this
  environment?** If not — if learnability here is dominated by seed-level optimisation
  variance — then the search problem should be re-framed (more seeds per candidate rather
  than more candidates; or a policy-side intervention such as exploration bonuses,
  curriculum on the release timing, or demonstration seeding) and the current framing
  (find a better reward) may be the wrong problem.
* **(Q3) What would a well-designed *next* experiment be**, given that (i) full pipelines
  cost hours, (ii) five selectors have failed, (iii) the oracle repair says the target is
  two localized edits away, and (iv) the prompt currently contains a self-contradiction
  that provably pays better for *not* finishing?

### 8.1 What we plan to do next (for critique)

1. Fix the v7 self-contradiction only ("the settled per-step payoff is exempt from the
   'all other components must be zero when settled' rule"), leaving all coefficients to the
   generator — a single, certain correction rather than a new hypothesis.
2. Instrument first, then spend: two seconds-long checks computed from a reward's
   *realised* component table (progress-integral vs its geometric bound; settled-term
   share/activation), used to screen generated candidates **before** training any of them.
3. One single-variable experiment that has never been run: `control_v1` + `probeD`'s
   closing-speed term, against a matched baseline, to isolate the one term that separates
   0 % from 73.3 %.
4. Only if (2)/(3) yield a candidate at ≥ 20/60, run the full pipelines for the method
   comparison.

### 8.2 What we have already ruled out (please do not re-propose)

* structure/dominance checks, trajectory-ordering checks, short-training rungs, the
  per-step advantage scale `A`, shaping:terminal ratio, hover wage, dock-dwell reward — as
  **selectors** (all measured, see §3.1);
* positional overshoot cliffs, extra gentleness penalties, proximity/state rewards and
  "near AND slow" funnels as **guidance** (§6.2);
* a mechanical gate built on `A > 0` (it passes arms that score 0/60);
* adding more evidence channels to the reflection (measured real 0/8 vs sham 1/8, §3.4);
* the "stronger generator" hypothesis (closed by the user);
* "make selection cheaper by training fewer steps" (measured: biased rungs, §3.1).

---

## 9. Artifact index (for whoever runs the next experiment)

* Entry points: `NEXT_SESSION.md`, `SESSION_STATE.md` (the full resume document),
  `runs/env_007/RESEARCH_LOG.md` (the research index), `HANDOFF_QUESTIONS.md`.
* Per-stage protocol + result pairs: `runs/env_007/*_PREREGISTRATION.md` and
  `runs/env_007/*_FINDINGS.md` (`LADDER_FINDINGS`, `REPAIR_TEST_FINDINGS`,
  `REPAIR_LOOP_FINDINGS`, `ADVANTAGE_PROBE_FINDINGS`, `ABLATION_FINDINGS`,
  `PILOT_TERMINAL_RULE`, `OVERSHOOT/APPROACH_ABLATION_PREREGISTRATION`,
  `RIDGE_WIDTH/RIDGE_RECOVERY_PREREGISTRATION`, `V7_PROMPT_PREREGISTRATION`).
* Training-free instruments: `analyze_terminal_dominance.py`,
  `trajectory_ranking_check.py`, `advantage_scale_probe.py`, `check_settled_stream.py`,
  `probe_v7_shape.py`, `analyze_native_reward.py`, `compare_shaping_scale.py`,
  `probe_metric_correlation.py`, `probe_progress_integral.py`.
* Raw runs, prompts (`prompts/eureka_01_initial_reward_v1..v7.md`) and configs are all on
  disk; the code and the full research log are published at
  `https://github.com/gcrxgcrx/fragile-cargo-dock`.


---

# ADDENDUM (same session, after the external review was received)

Three experiments were run in response to the review. All numbers below are measured; raw
JSON and the pre-registrations are in the repository.

## A1. Single-variable isolation of the one term that separates 0 % from 73.3 % — DONE

`runs/env_007/CLOSE_SPEED_ISOLATION_PREREGISTRATION.md` + `..._FINDINGS.md`.
`C0` = `control_v1`; `C1` = `C0` + the observation-only closing-speed penalty only
(verified: `max |C1 - probeD| = 0` and `max |C0 - control_v1| = 0` over 27 trajectories).
1.2M steps, paired training seeds 0/1/2, all scored on the fresh block **37000-37059**.

| arm | seed | success | dock | P(success\|dock) | max consecutive stable steps (mean/median) |
|---|---:|---:|---:|---:|---|
| C0 | 0 | **0/60** | 0.90 | 0.000 | **0.83 / 1** |
| C0 | 1 | **48/60** | 0.87 | 0.923 | — |
| C1 | 0 | **46/60** | 0.87 | 0.885 | **8.03 / 10** |
| C1 | 1 | **26/60** | 0.70 | 0.619 | **4.93 / 5** |
| C1 | 2 | **0/60** | 0.13 | 0.000 | — |

* The mechanism is unambiguous: `C0` docks in 90 % of episodes and **never holds the settled
  condition for even one step on average**; 0 of its 54 docked-and-failed episodes got within
  three steps of success. The added term moves that statistic to 5-8 steps.
* **But the effect does not survive pairing**: seed 1 reverses (48 vs 26 in the control's
  favour), and pooled over 120 episodes C1 72 vs C0 48 is Fisher **p = 0.55**. At 1.2 M steps
  with one training seed per arm, seed variance is comparable to the effect size.
* Consequence for design: **>= 3 training seeds per arm, paired comparisons only.** The
  review's request (pair on training seed) was implemented and it is what exposed this.

## A2. v8 prompt revision (fixing the self-contradiction) — form fixed, ceiling unchanged

`make_prompt_v8.py`, `prompts/eureka_01_initial_reward_v8.md`,
`runs/env_007/V8_PROMPT_PREREGISTRATION.md`. Three edits: the one-off event is demoted to
optional (with the clip arithmetic stated); the settled per-step payoff must not be switched
off by the candidate's own event; the "all other components must be exactly 0 when settled"
rule is narrowed to persistent *state* bonuses and the progress term must be signed.

* **8/8 candidates valid; 7/8 now pay per step on a settled state *after* their own success
  event fired** (the v7 family: 0-1/8). The rule was absorbed.
* One candidate was trained (1.2 M, seed 0) and scored on fresh block **38000-38059**:
  **4/60 = 6.7 %**, `dock_entered` **0.70**, and — for the first time in this project — its
  **progress term is bounded at 14.04 points/episode** where `v7_cand_01` accumulated **2747**,
  with the settled payoff now the dominant term (share 0.761).
* **It still does not settle**: max consecutive stable steps **2.42** (median 1), 2 of 38
  docked-and-failed episodes reached 7-9 steps.
* Its own gentleness term is present and worth **-1.73 points/episode**; the hand-written
  arm's equivalent term is what produces an 8-step hold.

**Conclusion:** the prompt is no longer the binding constraint on the reward's *form*
(shape and boundedness are now right). The binding constraint is the **effect size /
calibration** of the term that governs settling. Writing "penalise closing speed" is not the
same as writing a penalty strong enough to change the behaviour.

## A3. Early-training signal (not a selector)

The in-training `dock_settled_hold` activation rate (readable after ~300 k steps) orders the
runs loosely by final outcome: 0.0409 → 48/60, 0.0293 → 46/60, 0.0131 → 26/60, 0.0085 →
0/60, 0.0024 → 0/60. It is **not a selector** (the ordering is not strictly monotone), but it
is a candidate **early-stopping / seed-budgeting** signal, which is a different and defensible
use.

## A4. What we did NOT do, and why

**We did not start the full pipeline (CREATE vs EUREKA, 10 x 3 M).** Reasons, all measured:
1. the pre-registered S5 rule is met only literally — the paired comparison reverses on one
   of two seeds and the pooled difference is p = 0.55, so the effect it licenses is not
   established at the seed level;
2. every LLM arm, across v5/v7/v8 and both prompts, lands in the same 0-7 % band while
   A1 shows the missing capability is a *calibration* problem the pipeline does not address
   (its operator was measured inert: real 0/8 vs sham 1/8);
3. a pipeline run costs ~1.5 h (population method) to ~5 h (single lineage) and its most
   likely outcome is a second "both 0/20", which we already have.

**Open question for the reviewer:** given that the *form* is now right and the *calibration*
is wrong, is the next move (a) an explicit coefficient-calibration stage in the operator
(a sweep over the settling/gentleness scale, which the earlier ridge findings suggest is
non-monotone), (b) a multi-round loop (the one untested hypothesis), or (c) accepting that
the hand-written calibration must be supplied and re-framing the study around selection
between calibrated candidates?


---

# ADDENDUM 2 (06:25) — the seed-budgeted replication, and what it does to Q1/Q2

Response to the first addendum's open question, and a correction of the framing.

## B1. The isolation effect does not survive seed replication

`runs/env_007/SEED_REPLICATION_{PREREGISTRATION,FINDINGS}.md`. `C0` = hand-written
`control_v1`; `C1` = `C0` + one observation-only closing-speed penalty; 1.2 M steps;
**training seeds 0-3**; all scored on the fresh block 38000-38059:

| arm | s0 | s1 | s2 | s3 | mean | sd |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 0 | 43 | **54** | 35 | **33.0/60** | 23.3 |
| C1 | **45** | 21 | 0 | 13 | **19.8/60** | 18.9 |

Paired differences +45, −22, −54, −22; mean −13.2; 95 % paired t-interval **[−79.5, +53.0]**;
pre-registered verdict **N (null)**. The +77-point single-seed contrast was one draw.

## B2. The larger finding: seed class, not reward class

The **hand-written** control scores {0, 43, 54, 35} across four seeds of the *same* reward and
hyperparameters. Measured on the same block, across budgets:

| arm / seed | 1.2 M | 3.0 M |
|---|---:|---:|
| C0 seed 0 ("bad") | 0/60 | **2/60** |
| C0 seed 1 ("good") | 48/60 | **51/60** |

and from the existing record the same arm at 0.6 M scored 39/60 and at 1.0 M 22.2 % (seed
range 0-65 %). So a seed's outcome class is set early and persists; it is not slow learning.
**Consequences:**

1. the question "why do LLM rewards score 0-5 % while hand-written ones score 40-77 %" is
   partly mis-posed — at 1.2 M the *same* hand-written reward spans 0 % to 90 % across seeds,
   so LLM and hand-written samples come from overlapping distributions;
2. the project's single-seed selection rule preferentially keeps **lucky draws of the class
   that cannot succeed** and discards unlucky draws of the class that can — a measured
   pathology of the current pipeline design, independent of either method;
3. a cheap 1.2 M run can **rule a candidate out** (0/60 at 1.2 M stayed ~0 at 3 M in both
   seeds tested) but cannot **select a winner**; confirming a winner needs >= 3 training seeds
   with the seed as the unit of analysis.

## B3. What this does to the review's Q1 and Q2

* **Q1 ("what is the missing ingredient?")** — for the failing family the missing *capability*
  is settling, and one observation-only term supplies it in a controlled test (max consecutive
  stable steps 0.83 -> 5-8). But supplying it does not beat seed noise at 1.2 M, and scaling
  its coefficient up destroys locomotion (0/60, dock 0.00 at 15x and 50x). So the answer to Q1
  is not "a missing term": it is **an optimisation property — whether the training run lands in
  the basin where the settled behaviour survives optimisation.**
* **Q2 ("can P(learn) be predicted from the reward?")** — the first addendum answered "only as
  a distribution over seeds"; this round sharpens it: the *class* (0 %-class vs 50 %-class) is
  already visible in one 1.2 M run and is stable to 3 M, so **cheap runs are valid for
  rejection and invalid for selection**. That is a different (and usable) statement than
  "training-free selectors fail".
* **Consequence for the full pipeline:** the CREATE/EUREKA configs select by a single-seed
  score. Running them at 10 x 3 M would compare two methods inside a regime where the
  hand-written reference scores 2/60 on one seed and 51/60 on another. **The selection rule
  needs fixing before the method comparison is meaningful** — otherwise the headline number
  measures the seeding, not the method.


---

# ADDENDUM 3 (06:55) — the seed x budget interaction, and the corrected design rule

The final experiment of the session tested whether a 1.2 M run predicts a seed's class at 3 M
(`runs/env_007/SEED_REPLICATION_PREREGISTRATION.md`, addendum section). **Prediction falsified
for one arm, confirmed for the other:**

| arm / seed | 1.2 M | 3.0 M (same block 38000-38059) |
|---|---:|---:|
| C0 seed 0 | 0/60 | 2/60 |
| C0 seed 1 | 48/60 | **51/60** |
| C1 seed 0 | **45/60** | **3/60** (dock 0.88) |
| C1 seed 2 | 0/60 (dock 0.02) | 0/60 (dock 0.92) |

* one cell out of four survives 3 M (51/60); the other three score 0-3/60;
* both arms **dock** at 3 M (0.88-0.92) and still do not settle — so the failure at 3 M is a
  settling failure, not an exploration failure;
* `C1` seed 0 shows the collapse directly: 45/60 -> 3/60 with 3x the training, docking
  unchanged.

**Corrected design rule (replaces the one proposed in ADDENDUM 2):**

1. **No cheap screen for selection exists.** A 1.2 M run cannot tell whether a candidate will
   work at 3 M (C1 seed 0 looked like the best cell at 1.2 M and is among the worst at 3 M),
   and a 3 M run cannot be read from a 1.2 M run. Selection needs **>= 3 training seeds at the
   budget you intend to report**.
2. **Report the distribution, never a cell.** Every 3 M method/candidate number in this project
   is one draw from {0, 0, 2, 3, 51} out of 60 measured across four cells of two *reward files
   that differ by one term*. A single-seed 3 M number is not a property of the reward.
3. **The pipeline's selection metric is the thing to fix first.** CREATE/EUREKA both rank
   candidates by `mean_eval_reward` from one training seed. In the regime measured here that
   ranking is dominated by the draw, so a method comparison run on it measures seeding, not
   method. This is a concrete, cheap, and pre-registrable fix (rank by the *median of >= 3
   training seeds*, or by the settling statistic below) that does not require a stronger
   generator, new evidence channels, or a better prompt.
4. **The one signal that does track the outcome is a policy-behaviour statistic, not a reward
   property:** the realised per-episode contribution of the settled-state term during training
   (readable at 1.2 M, no extra cost). Measured: 19 -> 0/60, 68 -> 0/60, 97 -> 26/60, 140 ->
   46/60, 170 -> 48/60, 272 -> 51/60, with one inversion (380 -> 13/60). It is not a selector,
   but it is a usable **early-stopping / seed-prioritisation** signal, and it explains why the
   five *training-free* probes failed: they measure the reward function, this measures the
   policy's trajectory under it.
