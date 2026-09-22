# RESEARCH LOG — env_007 (`FragileCargoDock-v0`), baseline → v7

A reading-order index of the reward-search study on this environment. **Every number
below is measured and reproducible from the JSON/MD artifacts committed next to it.**
The per-stage detail lives in the pre-registration + findings pairs listed under each
stage; this file only fixes *what was asked, what was measured, and what it killed*.

Two companion documents are essential and are kept at the repo root:

| file | role |
|---|---|
| `SESSION_STATE.md` | the resume document: environment physics, harness facts, every established measurement, the falsified errors, and the warnings |
| `NEXT_SESSION.md` | the entry point after a context reset, plus the standing hard constraints (API keys, `deepseek-flash` only, the commit-limit trap, held-out seed blocks) |
| `HANDOFF_QUESTIONS.md` | the P1–P5 problem statement and the scope split (paper-reporting issues are out of scope) |

---

## 0. What the task is, and what the evaluation is

Top-down Box2D warehouse: a cart with **no brake** must push a fragile crate through a
narrow partition opening into a marked docking bay. Success requires the crate to be
**inside the dock tolerance ∧ aligned (<30°) ∧ slower than 0.05 m/s for 10 consecutive
steps**; the environment terminates the episode the moment that holds. Three hard
failures also terminate: 3 hard collisions, crate out of bounds, cart out of bounds.

Method under test: **CREATE** (called DERES in the paper's tables). Baseline:
**EUREKA-style population search** (`pipeline/run_eureka_population.py`).
**Scope: experiments only** — the paper is not ours, and paper-reporting issues are
listed in `HANDOFF_QUESTIONS.md` §6 so they are not re-litigated.

---

## 1. The baseline: the environment's own (native) reward

`runs/env_007/CALIBRATION.md` · harness `run_fragilecargo_baseline.py`

| policy | budget | fresh-60 success |
|---|---:|---:|
| random | — | 0 % |
| scripted heuristic pusher | — | 50 % |
| **PPO on the native reward, bypassing the wrapper** | 3.0 M | **96.8 %** |
| **PPO on the native reward, through the wrapper, clip 20** | 1.2 M | **90.0 %** |
| native reward, through the wrapper, clip 600 | 1.2 M | 80.0 % |

*The native reward was itself debugged by hand* (§3 of `CALIBRATION.md`): a
per-step contact bonus created a trapping local optimum (cart leaning on the jammed
crate for +5/episode), was replaced by an impulse-proportional cost, which then
destroyed exploration, which revealed the missing `approach_cargo` potential term
(0 % → 30–100 % across seeds). `normalize_reward` is the decisive training knob:
γ = 0.999 alone gives **0 %**, γ = 0.999 + reward normalisation gives **80 %**.

Native reward weights (masked from the generating LLM): `approach_cargo` 1.0/m,
`progress` 1.0/m, `dock_enter` +5 once, `roughness` −0.02 × peak impulse,
`action_cost` 5e−4, `time_cost` −0.002/step, `hard_hit` −0.5, `success` +300,
`failure` −100.

## 2. Harness validation — the harness is not the bottleneck

`runs/env_007/PILOT_TERMINAL_RULE.md` · `runs/env_007/ABLATION_FINDINGS.md`

* The native reward through `RewardOverrideWrapper` at clip 20 still scores **90 %**;
  raising the clip to 600 made the best LLM candidate *worse*. So the per-step clip and
  the wrapper are **not** what breaks the generated rewards.
* The observation-only contract **can** express a solvable reward (see stage 3).

## 3. The hand-written ceiling: what a working reward looks like

`runs/env_007/ABLATION_FINDINGS.md` §, `runs/env_007/LADDER_FINDINGS.md` §1

| arm | contents | fresh-60 |
|---|---|---:|
| `control_v1` | incremental crate progress + cart approach + `+20`/step settled stream + boundary guard | **0 %** |
| `probeA` | `control_v1` + a true contact-impulse penalty (reads `info`) | 40.0 % |
| `probeB` | per-step stream replaced by the native one-off terminal event (reads `info`) | 45.0 % |
| `probeD` | `+` observation-only closing-speed ("gentleness") penalty | **73.3 %** |
| `probeE` | obs gentleness + a module-state one-off success event via `obs[18]`, no `info` | **65.0 %** |
| `probeC` | true impulse + event form | **98.3 %** |

Also measured: the LLM search's historical best before this session was **5/60 = 8.3 %**
(a pilot candidate that collapses to 0/60 at full budget), and CREATE v2 / EUREKA v2 at
10 × 3 M both landed at 0–1/60.

## 4. The prompt ladder — the scaffold rules *I* added were the leak

`SESSION_STATE.md` §3e

16 candidates per level, generation only, same context/model/temperature:

| check | L0 (interface only) | L1 (paper's own prompt) | L2 (my scaffold) |
|---|---:|---:|---:|
| writes a one-off terminal event | 1/16 | 0/16 | **16/16** |
| writes a gentleness term (gap > 0) | 0/16 | 2/16 | **15/16** |
| all four structural checks | 0/16 | 0/16 | **11/16** |

L0 ≈ L1, so the leak is specifically the rules added on top of the paper's prompt.

## 5. The selector dead-ends (each one measured, each one rejected)

| stage | tool | question | outcome |
|---|---|---|---|
| 5a | `analyze_terminal_dominance.py` | do structural checks predict success? | **No.** 11 scaffold-compliant LLM rewards, all **0/60**; eight further candidates spanning every check, all 0/60 (`SESSION_STATE.md` §5). |
| 5b | `trajectory_ranking_check.py` | does the reward order reachable trajectories? | **No.** `control_v1` scores **0 %** yet ranks success above every failure with accuracy **1.000** and separation 199.6 (`SESSION_STATE.md` §3f). |
| 5c | `rung_analysis.py` | is a short (0.6 M) training rung a valid selector? | **No.** ρ = **−0.486** vs 1.2 M success, seed sd 0.236 > candidate spread 0.149; a component-activity readout that separates 14/14 at 1.2 M has recall 0.40 at 0.6 M. The 1.0 M rung: ρ = +0.371 and it certifies the 0 %-at-1.2 M `control_v1` in 3/3 seeds → rejected as pre-registered. |

## 6. Why they fail — mechanism, measured

`runs/env_007/LADDER_FINDINGS.md` §2 · `SESSION_STATE.md` §3h

Every arm that works puts ~96 % of its reward mass in a **positive, sparse (one-off or
stream) term that is actually reached**; every failing LLM arm puts **0 %** there. Its
mass sits in a penalty, in a **farmable dense term** (`L0/cand_02` collects +311 per
episode and still never docks), or nowhere at all (`L2/cand_05`'s reward is identically
0 on the trajectory its own policy produces → no gradient). Two named failure modes.

Also measured: ranking correctness ≠ learnability, twice — `L0/cand_02`'s alignment term
is **inverted** (`1 − |obs[10]|` while `DOCK_ANGLE = 0`), so its 94 %-of-mass term is
maximised by holding the crate perpendicular; the one-subscript fix moves the ordering
check 0.038 → **0.702** and the arm still trains to **0/60** (`ALIGN_FIX_PREREGISTRATION.md`).

And: **longer training can destroy a learned behaviour** — `control_v1` scores 39/60 at
0.6 M (seed 0) and 0/60 at 1.2 M, while all five hand-designed probes improve.

## 7. The repair study — the project's main positive result

`runs/env_007/REPAIR_TEST_PREREGISTRATION.md` · `REPAIR_TEST_FINDINGS.md` ·
`RECIPE_REPLICATION_PREREGISTRATION.md` *(oracle-authored: it measures the size of the
target, not its discoverability)*

Two localized edits take two independent LLM candidates from **0/60** to **23/60 = 38.3 %**
each (one seed reached **43/60 = 71.7 %**):

1. the candidate's **own delta-progress coefficient ×50** (12 → 600) — makes *acting*
   profitable: `dock_entered` 0.00 → 0.97, goal distance 4.148 m → 0.140 m, still 0/60
   because nothing pays for settling;
2. a **dense per-step payoff on the instantaneous success predicate** (inside ∧ aligned ∧
   slow) — makes *settling* profitable. Alone it does nothing (`r06`, `g01_stream` are
   bit-for-bit equivalent to their copy controls, because the predicate is never reached).

Controls: adding gentleness **hurts** (0/60, dock 0.00); raising the candidate's own speed
penalty ×100 makes it worse (0/60, predicted in advance by the advantage probe);
the untouched copy reproduces the base exactly.

## 8. The evidence channel is not the bottleneck

`runs/env_007/REPAIR_LOOP_PREREGISTRATION.md` · `REPAIR_LOOP_FINDINGS.md`

The pipeline's own reflection — verified to contain the decisive
`success_event` activation evidence — was given to the operator, with a permuted-table
sham control, 8 repairs per arm: **real 0/8, sham 1/8**, Fisher p = 1.0000. The ceiling
is the **operator**, not the evidence, so new channels (including P5) are predicted
worthless on this environment until iteration or a different prompt shape is tested.

## 9. Advantage scale: a diagnostic, not a selector

`runs/env_007/ADVANTAGE_PROBE_PREREGISTRATION.md` · `ADVANTAGE_PROBE_FINDINGS.md`

`A` = mean per-step generated return over seven scripted controllers − that of `idle`,
computed with **no training**. It answers a real question — `A ≤ 0` means *inaction is
optimal*, which is the whole v5/v7 immobile family — and it predicted, before their
results were read, which arms would move the cart (the four containing the ×50 edit flip
`A` from −0.259 to +0.11…+0.14 and are the four whose policies dock) and that scaling the
candidate's own speed penalty ×100 would be catastrophic (`A` = −12.8).

It is nevertheless **rejected as a selector**: AUPRC 0.413 (prevalence 0.357), and the
three *failing* L0 candidates hold the largest `A` of the pool (+0.39, +0.32, +0.17),
above every working arm — a dense farmable term pays more while acting than while idling.

## 10. v7 — removing the scaffold rule that forbade the ingredient

`runs/env_007/V7_PROMPT_PREREGISTRATION.md` · generator `make_prompt_v7.py` ·
diff `runs/env_007/v7_prompt.diff` · prompt `prompts/eureka_01_initial_reward_v7.md`

v5 (lines 112–119) banned a per-step payoff on the success state and its self-check ④
forced the model to rewrite any such payoff. That ban was justified by a real measurement
whose attribution was wrong: `control_v1` ground in the dock for 43 steps because its
*approach* term was far too weak (`A = −0.26`), not because of the stream. v7 replaced the
ban with a conditional rule requiring **both** a one-off event and a per-step settled
payoff, and inverted self-check ④.

| v7 outcome | value |
|---|---|
| candidates that pay per step on a settled state | **8/8** (v5 family: 0/2) |
| candidates with `A > 0` | 4/8 (v5 family: 1/8) |
| first unedited LLM candidate ever to enter the dock | `cand_01`, `dock_entered` **0.18** (all 35+ earlier ones: 0.00) |
| honest rate | **1/8 hits, 3/60 = 5.0 %**, one-sided Fisher p = 0.333 vs the 0/16 baseline |

**Reading:** the prompt fix is real but insufficient; the family splits cleanly on `A`,
and the mechanic that carries the missing ingredient is still not produced.

## 11. The difficulty is two-factor, and settling is near-coin-flip

`SESSION_STATE.md` §5.7–§5.8 · `runs/env_007/RIDGE_WIDTH_PREREGISTRATION.md`

Single-axis sweep of the one coefficient the working recipe sets, same fresh block
35000–35059, then seeds 1–2 added:

| coefficient | seed 0 | seed 1 | seed 2 | mean | `dock_entered` per seed |
|---|---:|---:|---:|---:|---|
| ×50 (600) | 16/60 | — | — | (16/60) | 0.90 |
| ×75 (900) | 0/60 | 4/60 | 0/60 | **2.2 %** | 0.07 / 0.48 / 0.00 |
| ×100 (1200) | 31/60 | **43/60** | 0/60 | **41 %** | 0.95 / 0.78 / 0.95 |

* **Reachability is set fairly reliably by the coefficient**; **settling is the binding,
  high-variance step** — ×100 seed 2 docks in 95 % of episodes and scores **0/60**.
* Therefore **single-seed candidate comparisons are one lottery draw**: report
  `dock_entered` and success **separately** and state the seed count.
* The advantage probe is **blind** to this: `A` is smoothly monotone in the coefficient
  (+0.141 / +0.331 / +0.521 / +1.281) while the outcomes are jagged, ranking the worst
  arm (×200, 0/60) highest. A gate built on `A > 0` would pass ×75 and ×200.
* Decision recorded (user): **no further multi-seed confirmation runs**; the coefficient
  is **deliberately left to the LLM/search**, with the measured caveat that scale is a
  first-order, non-monotone variable here, so a *sampling* loop suits it better than a
  gradient-following one.

## 12. Guidance ablations: nine single-axis "expert" edits destroy a working reward

`runs/env_007/OVERSHOOT_ABLATION_PREREGISTRATION.md` ·
`runs/env_007/APPROACH_ABLATION_PREREGISTRATION.md`

Against a matched **16/60, dock 0.90** baseline in the same fresh block:

| the idea | arm | fresh-60 | dock |
|---|---|---:|---:|
| make overshoot very negative | `o01_overshoot` (−20/step cliff) | **0/60** | 0.00 |
| same, 10× harder | `o04_overshoot_x10` (−200/step) | **0/60** | 0.00 |
| add a gentleness / closing-speed penalty | `o02_gentle` | **1/60** | 0.35 |
| overshoot cliff + gentleness | `o03_both` | **0/60** | 0.00 |
| reward approaching, state (proximity) form | `p01_proximity` | **0/60** | 0.02 |
| approaching, delta form ×4 stronger | `p02_progress_x200` | **0/60** | 0.72 |
| approaching, "near AND slow" funnel | `p05_funnel` | **0/60** | 0.07 |
| proximity + gentleness / ×200 + gentleness | `p04`, `p03` | **0/60** | 0.10 / 0.53 |

Mechanism (measured, not interpreted): with no brake the only route to the dock is to push
and let drag stop the crate, so arriving trajectories pass near the far edge; a positional
cliff at −20/step makes the **approach itself** unprofitable, `A` flips from +0.141 to
−0.253, and the optimum becomes *not approaching* (dock 0.90 → 0.00).

## 13. Open, pre-registered, not yet run

`runs/env_007/RIDGE_RECOVERY_PREREGISTRATION.md` · costs `runs/env_007/COMPUTE_LEDGER.md`

Seed each method's operator with an off-ridge candidate whose failure cause is **known**
(`p02_progress_x200`: arrives, cannot settle; `o01_overshoot`: avoidance) and ask whether
its diagnosis recovers delivery. EUREKA side = `build_reward_reflection` +
`materialise_reward(mode="edit")`; CREATE side = `pipeline/run_04_build_iteration_context.py`
then `pipeline/run_05_reward_revision.py`. 2 operators × 2 seeds × 4 repairs = 16 trainings.

---

## 14. Addendum — measurements made after §13 was written (v7 scoring, 2026-09-21)
Three training-free diagnostics were added at the end of the v7 round. They are recorded
here because they bear on what any v8 prompt must say; they are **not** yet written into
any pre-registration.

**14a. Under the harness clip, a one-off completion event is worth exactly nothing.**
`probe_v7_shape.py` replays a reward on the scripted success trajectory (release_0.25,
303 steps) and on the same trajectory truncated one step before completion plus 40 steps
that keep the settled predicate true without completing it, at clip 20:

| candidate | terminal step (raw → clipped) | cumulative to completion | truncated + 40 farm steps |
|---|---:|---:|---:|
| `cand_00` | 320.0 → **20.0** | 96.7 | **876.7** |
| `cand_01` | 300.0 → **20.0** | 3083.7 | **3263.7** |
| `cand_02` | 320.0 → **20.0** | 333.4 | **1113.4** |

The one-off's +300 is clipped to the same value as a single settled step, so it adds
**zero**; and because v7 candidates gate the per-step payoff on `not _PAID` (or zero it
once the event fires), *not completing* is worth ~+780 more than completing for `cand_00`.

**14b. The native reward's cumulative shaping is bounded by geometry.**
`analyze_native_reward.py` decomposes the native reward on 27 scripted trajectories
(9 controllers × 3 seeds): the shaping total (approach + progress + dock_enter + action +
time) spans **−3.2 … +8.6**, and on a successful 303-step trajectory it is **+8.58**
against a terminal **+299.84** — a ratio of about **1:35**, i.e. near-sparse. Both
telescoping terms (`approach_cargo`, `progress`) have totals bounded by geometry, so
circling, jittering and hovering cannot collect anything extra.

**14c. LLM candidates sit 2–4 orders of magnitude away from that ratio.**
`compare_shaping_scale.py`, same trajectory (shaping = every step except the terminal one,
clipped; "last40/step" = what a hovering policy collects):

| reward | shaping | terminal | ratio | last40/step |
|---|---:|---:|---:|---:|
| native (reference) | 8.45 | 300 | **0.028** | 0.028 |
| `probeE` (76.7 %) | 9.17 | 20 | 0.458 | 0.114 |
| `probeD` (73.3 %) | 184.2 | 20 | 9.2 | 4.49 |
| v7 `cand_00` | 76.7 | 20 | 3.8 | 4.11 |
| v7 `cand_02` | 313.4 | 20 | 15.7 | 4.57 |
| v7 `cand_01` (= the 5 % hit) | 3063.7 | 20 | **153** | 6.78 |
| `r09` (oracle repair, 38.3 %) | 323.9 | 20 | 16.2 | 5.02 |
| `q02` (×100, 41 %) | 551.2 | 20 | 27.6 | 5.24 |

Ordering by ratio tracks the measured `dock_entered`/success ordering: the closer a
reward's shaping-to-terminal ratio is to the native reward's, the better it learns.
Reported here as an observation with n = 1 trajectory per reward; it has not been
pre-registered and no causal claim is made yet.

---

## 15. File map

| category | files |
|---|---|
| entry / resume | `NEXT_SESSION.md`, `SESSION_STATE.md`, `HANDOFF_QUESTIONS.md`, this file |
| environment | `custom_envs/fragile_cargo_dock_env.py`, `envs/env_007/`, `envs/env_007/task_spec_anonymized{,_v2}.yaml` |
| harness | `training/train_sb3_wrapper.py`, `training/reward_wrapper.py`, `configs/env007_*.yaml`, `runs/env_007/train_queue.ps1` |
| baseline | `run_fragilecargo_baseline.py`, `runs/env_007/CALIBRATION.md`, `runs/env_007/{baseline,ac3_s0..s3,calib_*,confirm_s*}/` |
| prompts | `prompts/eureka_01_initial_reward{,_v2..v7}.md`, `prompts/eureka_02_reward_edit*.md`, `make_prompt_v7.py` |
| generation-only tools | `pilot_generate_only.py`, `pilot_train_existing.py`, `pilot_ab_summary.py` |
| training-free diagnostics | `analyze_terminal_dominance.py`, `trajectory_ranking_check.py`, `advantage_scale_probe.py`, `check_settled_stream.py`, `measure_release_ballistics.py`, `component_share_report.py`, `probe_v7_shape.py`, `analyze_native_reward.py`, `compare_shaping_scale.py` |
| selection studies | `ladder_analysis.py`, `rung_analysis.py`, `mechanistic_readout.py`, `diagnose_ridge.py`, `eval_pool.py`, `eval_fresh_seeds.py`, `diagnose_control.py` |
| oracle repairs and ablations | `make_repair_variants.py`, `make_recipe_replication.py`, `make_align_fix_variants.py`, `make_overshoot_variants.py`, `make_approach_variants.py`, `make_ridge_variants.py`, `run_repair_loop.py`, `repair_loop_analysis.py`, `v7_analysis.py` |
| raw runs | `runs/env_007/{prompt_ladder,prompt_ladder_v7,prompt_ab,terminal_rule_pilot*,repair_test,align_fix_test,recipe_replication,repair_loop,overshoot_ablation,approach_ablation,ridge_width,rung_06m,rung_10m,ladder_train,v4_train,control_obs_only,passthrough_probe,ablation_probe,fragilecargo_create*,fragilecargo_eureka*}/` |

## 16. Standing constraints (violating these invalidates the work)

* `EUREKA_DEEPSEEK_API_KEY` for the generation/EUREKA flow, `DEEPSEEK_API_KEY` for CREATE.
  Keys are read from the environment only and are **never** written to a file. The
  reproduction scripts under `runs/env_001/ablation_eureka_feedback_v4/reproduction/`
  expect the key in the environment.
* Only `deepseek-flash` for generation; `DEEPSEEK_THINKING=disabled` must be exported.
* **Do not launch 8 trainings at once.** The binding resource is the Windows page
  file / commit limit, not RAM; workers then die during `import torch` with
  `WinError 1455` and the parent hangs at 0 CPU. Use `runs/env_007/train_queue.ps1`
  with `-MaxParallel 4` or lower.
* `runs/env_007/passthrough_probe/reward.py` and `runs/env_007/ablation_probe/*` read
  `info` and are diagnostics: never add them to a population, lineage or elite set.
* The 60 fresh seeds **30000–30059** are the honest evaluation block and must stay held
  out from any selection.

---

## 17. CREATE with the fixed spec + the v9 structure block (2026-09-22, after §16)

Pre-registration `CREATE_V9_PREREGISTRATION.md`, write-up `CREATE_V9_FINDINGS.md`, raw
`runs/env_007/fragilecargo_create_v9/`. `configs/env007_fragilecargo_create_v9.yaml` = CREATE
(`pipeline/run_iterative_experiment`, single lineage) with **two** changes against the run the
baseline numbers came from: `inputs.task_spec_path` -> `task_spec_anonymized_v2.yaml` (the geometry
the analyzer never saw, `PIPELINE_CONTEXT_DIAGNOSIS.md`) and
`inputs.reward_structure_context_path` -> `runs/env_007/CREATE_V9_STRUCTURE.md`, the nine native
reward terms with their weights, extracted verbatim from `prompts/eureka_01_initial_reward_v9.md`
by `tools/extract_v9_structure_block.py` and injected into the generator's and the reflection
agent's prompts.

10 rounds × 3 M steps, seed 0 (30 M env steps, same budget as EUREKA and as the v2 baseline).
Training score is the pipeline's 20-episode evaluation; **fresh** is the unused block
41000-41059, 60 episodes:

| round | training (20 eps) | fresh 41000 | round | training | fresh |
|---:|---:|---:|---:|---:|---:|
| 1 | 279.630 (18/20) | **55/60** | 6 | 309.545 (20/20) | **56/60** |
| 2 | 204.458 (13/20) | 25/60 | 7 | 294.207 (19/20) | **56/60** |
| 3 | 279.630 (18/20) | **55/60** | 8 | 309.545 (20/20) | **56/60** |
| 4 | 263.777 (17/20) | 51/60 | 9 | **309.773 (20/20)** | 53/60 |
| 5 | 309.545 (20/20) | **56/60** | 10 | 53.867 (3/20) | 1/60 |

**n = 10 rounds, mean 46.4/60, sd 17.6, best 56/60, 8 of 10 rounds ≥ 50/60.**

* CREATE's **round 1** was already delivery-level (55/60), against a baseline whose best round was
  17.694 in training and 0-1/60 fresh. Round 1's realised composition is `terminal_success` **95.97 %**
  of return against the native reward's ~96 % (`§14b`); the v2 baseline's round-1 mass sat in
  `crate_dock_alignment` (80.8 %), i.e. in a term that pays without completing.
* Rounds 5-8 hold at the ceiling (training 309.545, fresh 56/60), matching the best candidate of the
  full EUREKA-v9 run. Round 10 is the failure mode that matters: dock 0.80 but 1/60 — it arrives and
  cannot settle, the same signature as §11's settling variance.
* **n = 1 training seed**, and the comparison against `fragilecargo_create_v2` is confounded by a
  prompt revision that no config can restore (see the pre-registration §4). The structure block is
  supplied text, not something CREATE discovered.
* Three pipeline defects were found and fixed while running this, all independent of the result and
  all documented in the pre-registration §5: truncated environment cards were accepted as valid
  (the analyzer returns `finish_reason='length'` on ~half of calls on this environment; a 146-byte
  card was fed to the generator — now guarded by `llm.min_chars_env_card`), the subagent
  investigator raised `TypeError` on every call and was silently skipped in **every round of the v2
  baseline too**, and on resume `solved_seen` was reconstructed by a substring test that could never
  match.
* Protocol note: the run was stopped by CREATE's own stop rules twice (after rounds 2 and 4), and a
  third stop after round 7 exposed that the new `--no-early-stop-all` override had been placed
  before the config read and was inert. Rounds 8-10 ran with adaptive stops disabled; scores are
  unaffected (each round is scored by its own training run).

---
