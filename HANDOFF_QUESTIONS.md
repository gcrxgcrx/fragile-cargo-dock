# Handoff: open problems in the DERES/CREATE reward-search experiments on `FragileCargoDock-v0`

Written 2026-09-20. Everything below is measured; hypotheses are labelled as such.
Repository: `D:\Code\python\research\form_github\expert-reward-agent`
Interpreter: `D:\Code\python\research\llm_env_310\Scripts\python.exe`

## SCOPE (read first)

We are **only running experiments on `FragileCargoDock-v0`**; the paper is not ours.
The questions below are therefore split:

* **In scope — section 4, P1–P5.** Everything about this experiment: how the pipeline
  produces evidence, how the prompt is specified, whether this environment can
  differentiate the two search methods *within itself*, whether a cheap selector is
  salvageable, and how to extend the evidence channel.
* **Out of scope — section 5.** Defects in the paper's own reporting (statistic units in
  the env_001 ablation table, SD convention, naming). Listed only so they are not
  confused with the experimental questions; they go to the paper's author.

For **P3** specifically: the *cross-environment* claim ("EVSI predicts the
CREATE-minus-population gap") needs ~20 tasks and is a paper-level programme. The
**in-scope** part of P3 is only: *can this one environment, within itself, show that
diagnostic evidence buys anything?* That is a within-task paired question and needs no
benchmark. P5's proposal is the within-task version of the same test.

---

## 0. What the method is, and what the paper claims

The method in the code is called **CREATE**; the figures/tables and the blueprint
docs call the same thing **DERES**. Same pipeline (`pipeline/run_iterative_experiment.py`),
same runs. Paper drafts: `paper_rewriting_output/{iscsic2026_session7,final_paper_v2}`.

CREATE = single-lineage reward repair: generate one reward, train, assemble structured
evidence (per-component activation rate and magnitude share, episode length, training
curves), diagnose a primary failure hypothesis, apply a severity-gated **localized**
edit (L1 parameter tuning / L2 component refactoring / L3 structural redesign), keep a
best archive. The EUREKA-style population baseline is `pipeline/run_eureka_population.py`.

Published-style ablations on **LunarLander (env_001, threshold 200, 10 rounds x 1M, 5 seeds)**:

| arm | n | solved | best mean |
|---|---:|---:|---:|
| CREATE full | 5 | **5/5** | **228.98 ± 18.49** |
| score-only feedback | 5 | 3/5 | 174.69 |
| **EUREKA-style feedback** | 5 | **2/5** | 134.97 |
| unconstrained rewrite (no L1/L2/L3) | 5 | 0/5 | 114.21 |
| budget-matched independent multi-sample | 8 | 1/8 | −53.27 |

Fisher one-sided: 5/5 vs 1/8 → p = 0.005; vs 0/5 → p = 0.004; **vs 2/5 → p = 0.083
(not significant)**. So the "structured evidence beats EUREKA-style feedback" claim is
currently underpowered; the other two are solid.

---

## 1. The environment under study

`custom_envs/fragile_cargo_dock_env.py`, id `FragileCargoDock-v0`, **authored by us**.
Top-down Box2D warehouse; gravity 0; drag is the only way to slow anything.
`obs` Box(19) clipped ±2; `action` Box(2); 400 steps; dt = 1/30.

- Success: crate fully inside the dock (`DOCK_HALF - CARGO_HALF = 0.42 - 0.30 = 0.12 m`
  on both axes), heading error < 30°, crate speed < 0.05 m/s, **held 10 consecutive steps**.
  The episode **terminates immediately** on success.
- Failure: cart or crate leaves the field, or ≥ 3 hard impacts (peak normal impulse > 5 N·s).
- **Core mechanic: the cart has no brake and cannot slow the crate from behind.** It must
  release early and let floor drag carry the crate into the bay.

### Measured ballistics (`measure_release_ballistics.py`)

Coast distance obeys `(v0 − 0.05) / 2.0` (linear damping 2.0), dock at x = 2.6:

| release speed | coast | release must happen at | allowed position error | timing margin |
|---:|---:|---:|---:|---:|
| 0.2 m/s | 0.074 m | 2.526 | ±0.12 m | 18.0 steps |
| 0.3 m/s | 0.123 m | 2.477 | ±0.12 m | 12.0 steps |
| 0.5 m/s | 0.222 m | 2.378 | ±0.12 m | 7.2 steps |
| 1.0 m/s | 0.468 m | 2.132 | ±0.12 m | 3.6 steps |
| 2.0 m/s | 0.959 m | 1.641 | ±0.12 m | 1.8 steps |

The settling takes **21–55 steps (0.7–1.8 s)** after release. So the task is a ballistic
commitment, and pushing fast is the only thing that shrinks the timing margin. "Gentle"
and "well-timed" are the same problem here.

---

## 2. The harness (this matters)

`training/reward_wrapper.py` — every **generated** reward goes through
`RewardOverrideWrapper`:

- per-step clip to `reward_clip`, **default 20.0** (`training/train_sb3_wrapper.py:656`);
- the native reward is measured with `reward_fn = None`, i.e. it **bypasses the wrapper
  entirely** and is never clipped.

Training config for all arms below: `n_envs=6`, `gamma=0.999`, `normalize_reward=true`,
3M steps per candidate.

---

## 3. What is established (all figures: 60 **fresh** seeds 30000–30059)

### 3a. The harness is fine; the clip is not the problem

| policy | budget | fresh-60 |
|---|---:|---:|
| native reward, **bypassing** the wrapper | 3.0M | **96.8 %** |
| native reward, **through** the wrapper, clip 20 | 1.2M | **90.0 %** |
| native reward, through the wrapper, clip 600 | 1.2M | 80.0 % |

### 3b. The observation-only contract *can* express a solvable reward

All arms are 1.2M steps, seed 0, fresh-60, clip 20. Every arm is the same hand-written
"control v1" with exactly one thing changed:

| arm | change vs control v1 | fresh-60 |
|---|---|---:|
| control v1 | — (incremental progress + approach + a `+20`-per-step settled stream + boundary guard) | **0 %** |
| probe A | `+` true contact-impulse penalty, read from `info` | 40.0 % |
| probe B | per-step settled stream replaced by the native one-off terminal event (`info`) | 45.0 % |
| probe D | `+` **observation-only** closing-speed penalty, `−0.05 · contact · max(0, cart_fwd_speed − crate velocity along heading)` | **73.3 %** |
| probe E | probe D's gentleness **+** a one-off terminal event built from module-level state + `obs[18]` episode-boundary detection (**no `info` at all**) | **65.0 %** |
| probe C | probe A's true impulse **+** probe B's event form | **98.3 %** |

**Conclusion: the task is solvable by observation-only rewards, and the single decisive
ingredient is a signal that penalises *closing speed*, not proximity.** Without it the
policy pushes the crate through the dock (measured: spends 43.5 steps inside the dock,
only 4.8 of them below the speed threshold, 30/60 episodes reach 5–9 of the 10 required
settling steps and never complete).

### 3c. The LLM search has never produced a usable reward

| run | budget | best selection score | fresh-60 | fresh success |
|---|---|---:|---:|---:|
| EUREKA v2 (old prompt/spec) | 10 x 3M | +8.093 | +7.027 | **0/60** |
| CREATE v2 (old prompt/spec) | 10 x 3M | +17.694 | +7.716 | **1/60 = 1.7 %** |
| pilot cand_03 (dock geometry + terminal rule) | 1.2M | +34.884 | +31.30 | **5/60 = 8.3 %** |
| same reward, full budget | 3.0M | +4.152 | +4.00 | **0/60** |

The 8.3 % was a transient: at 3M the same reward drives the policy into parking just
outside the dock while the generated return climbs monotonically (100 → 126/episode).

### 3d. A 24-candidate structural ruler (no training needed)

`analyze_terminal_dominance.py --clip 20 --summary <rewards>` scores every reward ever
produced on synthetic states and scripted controllers:

| contract | n | reward at the dock ÷ reward parked just outside | episode-credit ratio |
|---|---:|---:|---:|
| old spec + old prompt | 20 | **1.02 – 1.24** | 0.025 – 0.031 |
| pilot (dock geometry + terminal rule) | 4 | **172 – 5123** | 0.289 – 1.278 |

The old twenty candidates made "docked" worth 2–24 % more than "not docked", and their
largest dock reward was exactly **20.00 = the clip value**: the old search never even
tried to signal a completion event. This is a ~1000x design change, measurable per file.

### 3e. Prompt compliance can be engineered — but compliance ≠ quality

Generation-only A/B, 16 candidates per arm, scored mechanically (no training):

| check | v2 | v3 | v3b | **v4** |
|---|---|---|---|---|
| writes a one-off terminal event | 0/16 | 6/16 | 16/16 | **16/16** |
| writes a closing-speed (gentleness) term | 0/16 | 16/16 | 16/16 | **16/16** |
| event beats the per-step residual | 0/16 | 2/16 | 3/16 | **16/16** |
| hover cannot out-pay docking | 6/16 | — | 5/16 | **15/16** |
| **all four** | **0/16** | 2/16 | 0/16 | **15/16** |

**Then we trained three of the v4 "all four" passers for 1.2M steps: all three scored
0/60.** Failure modes were concrete: one went out of bounds on 40/40 episodes (no
boundary guard at all); one entered the dock at **1.555 m/s** (needs < 0.05) because its
gentleness weight was copied verbatim at −0.05 and swamped by the progress reward.

So the mechanical gate measures **rule compliance**, not reward quality, and the
gentleness check in particular is *constant* across v4 candidates (they all copied the
same snippet), i.e. it has no discriminative power at all.

---

## 4. The actual problems I need help with

### P1. A "missing incentive term" is invisible to the evidence both methods receive

CREATE's evidence channel (`pipeline/run_reflection_agent.py`, `subagent_investigator.py`,
`run_04_build_iteration_context.py`) contains: per-component mean / abs_mean /
nonzero_rate (activation) / min / max / magnitude share, **mean episode length and its
delta between iterations**, termination breakdown, the training return curve, and the
retrieved expert cards. EUREKA's reflection contains the same class of things
(`run_eureka_population.py:175-230`).

The failure on this environment is **not** "a component has the wrong scale" — it is
"the reward has no term that penalises closing speed at all". Nothing in either evidence
channel can show a term that does not exist. Detecting it requires **trajectory-level
probing** ("at what crate speed does the episode enter the dock?"), which neither
method currently performs.

Compare LunarLander, where the paper's own case study is literally *"multiply the three
stability coefficients by 5"* and *"add an action-energy penalty"* — i.e. the defect sits
in components that **are** in the table. That is why the component evidence table pays
off there.

**Question:** what is the principled evidence construction for diagnosing an *absent*
incentive? Is counterfactual reward probing (evaluate the reward on scripted states /
scripted controllers, as `analyze_terminal_dominance.py` does) the right general answer,
or is there a better-established formulation?

### P2. Where is the line between "specifying the interface" and "handing over the answer"?

I made a design error: to fix the search I added a **closing-speed formula** and a
**one-off-event code skeleton** to the *shared* prompt used by both methods. That made
both methods "copy", so the method comparison became meaningless — and it is the main
reason this environment currently shows no CREATE-vs-EUREKA difference.

**Question:** is there a formal or at least operational criterion separating
(a) legitimate interface specification (e.g. "the per-step reward is clipped to ±20";
"the episode terminates on success"; "the observation has no impulse channel") from
(b) leaking the solution (e.g. giving the gentleness formula)? The whole experiment
design depends on this line.

### P3. Is `FragileCargoDock-v0` structurally incapable of showing the method's advantage?

I can state three conditions that must hold for each of the paper's three ablation
results to be reproducible:

- **A (repairability)** — good rewards are reachable from bad ones by a *sequence of
  small edits*, while random sampling has low hit probability → needed for
  "repair beats independent sampling".
- **B (preservation)** — the starting reward already contains several *correct*
  components that a rewrite tends to destroy → needed for "localized edit beats
  unconstrained rewrite".
- **C (evidence asymmetry)** — the defect is visible in CREATE's evidence but not in
  EUREKA's → needed for "structured evidence beats score+means".

My assessment for this environment: **A and B probably hold, C does not.** The most
effective repair I found is literally a one-component addition (probe D: control v1 + one
term = 0 % → 73.3 %), which is exactly an L2 localized edit.

The sharpest structural difference I found is the **initial-reward distribution**:

| environment | initial scores per seed | threshold | best initial as % of threshold |
|---|---|---:|---:|
| LunarLander (env_001) | −70.4, −42.7, −17.9, −19.6, **+139.5** (mean −2.2, sd 82.0) | 200 | **70 %** |
| BipedalWalker (env_002) | 270.7, 280.5, 272.1, 290.0, 103.0 | 300 | **97 %** |
| **FragileCargoDock** | ≈ −1.9 … +5.1 | ≈250 (equivalent) | **≈2 %** |

A repair method needs something partially correct to repair. Here every naive reward is
equally bad for the same reason (one missing term), so the initial population is
degenerate: breadth sampling cannot help, and there is no component to re-scale.

**Question:** is "mass of the initial-reward distribution near the threshold" the right
predictor of when diagnosis-guided single-lineage search beats population sampling? What
is the correct, ideally *a priori* measurable property? And is there a principled way to
construct environments (or tasks) that provide C without being designed *for* CREATE —
since LunarLander/BipedalWalker provide it incidentally?

### P4. Is a cheap structural gate salvageable as a selector?

The motivation: at 0–5 % success rates, selecting candidates by `mean_eval_reward` over
20 fixed seeds is nearly information-free (CREATE's +17.694 on the training seeds fell to
+7.716 on 60 fresh seeds; the training seeds are also the ones the search selects on).
A probe-based gate costs seconds instead of 25 minutes and has ~1000x dynamic range.

Current status: it **correctly rejected** two candidates that then scored 0/60, but
**three candidates it accepted also scored 0/60**. So it has no demonstrated positive
predictive power. The checks are: hover-cannot-out-pay-docking (episode-credit ratio),
terminal form (spike vs median over 12 repeated calls on the settled state), gentleness
(same contact, two closing speeds), and a runtime smoke test.

**Question:** can such a gate be made predictively valid — e.g. by probing *scripted
controller* returns rather than synthetic states, or by requiring the reward to rank a
library of reference trajectories correctly? Or is this fundamentally not learnable
without training?

### P5. What is the minimal environment modification that creates condition C?

My earlier suggestion ("add time pressure") does **not** work, and I verified why:
`mean_episode_length` and `termination_breakdown` are present in **both** methods'
evidence channels (`run_reflection_agent.py:197`, `subagent_investigator.py:26`,
`run_eureka_population.py:191-195`), so a timeout failure is visible to both and cannot
differentiate them. It would only add noise (fewer successes → noisier selection).

To create C, the failure must be of the form "**an existing component has the wrong
scale**", so that activation rate / magnitude share exposes it. That requires the naive
reward to *naturally contain* the component that needs tuning — i.e. genuinely competing
objectives that every naive design will include but weight wrongly. **I do not currently
have a small change that provably produces this**, and I would rather ask than invent one.

**Question:** given the environment described in section 1, what is the smallest change
that makes the dominant failure mode "wrong scale of an existing component" rather than
"absent component"? Or is the correct move instead to extend the *method's* evidence
channel with trajectory probes (i.e. accept that this environment tests a different
capability)?

---

## 5. In-scope second-order issue

- **The environment is ours.** LunarLander/BipedalWalker are standard, so their results
  are not open to the "you designed it for your method" objection. Any conclusion drawn
  from `FragileCargoDock-v0` alone carries that threat to validity. This is an
  experiment-level concern and stays in scope.

---

## 6. OUT OF SCOPE for this experiment - paper-reporting defects only

These are recorded so they are not mixed into the experimental questions above. They
concern the paper's own tables, not `FragileCargoDock-v0`; they go to the paper's author.

- **The env_001 ablation table's statistic units do not match its own source data.**
  `runs/env_001/budget_matched_independent_v1/summary.md` records `num_samples: 10`,
  `best = 228.998` (which is **>= the 200 threshold, i.e. solved**), `mean = -40.071`,
  `solved = 1/10`. The paper table reports `n = 8`, `solved = 1`, `best_mean = -53.27`
  (which is the mean of the **first eight** samples only) and never reports `best`.
  Also, `best_mean` is not the same statistic in the two arms: for the sampling baseline
  it is the mean over **single-evaluation** samples, for CREATE it is the mean over
  **multi-evaluation lineage bests** (evaluations per lineage: 9, 10, 3, 6, 5 = 33 total).
  Under a best-of-budget reading, best-of-10-samples = 228.998 and CREATE's mean best =
  228.98 - the same number.
- **Standard-deviation convention**: `experiments.tex` reports `+/-16.54` (population SD,
  and says so) while `figures/paper/deres_main/tables/*.csv` report `+/-18.49` (sample
  SD) for the same five values.
- **Naming**: `CREATE` in both `main.tex` drafts and in all code comments vs `DERES` in
  `docs/DERES_*.md` and `figures/paper/deres_main/tables/*.csv`; no rename note exists.
- **Underpowered ablation**: CREATE 5/5 vs EUREKA-style feedback 2/5 gives Fisher one-sided
  p = 0.083. Adding three seeds (`run_ablation_eureka_feedback_v4.sh`) would give 8 per arm
  where 8/8 vs ~3/8 gives p ~ 0.013. This is env_001 work, not `FragileCargoDock-v0` work.

## 7. Errors I made and then falsified myself (so they are not re-litigated)

1. "The per-step clip is the blocker" → falsified: native through the wrapper at clip 20
   scores 90 %; raising the clip to 600 made the best LLM candidate *worse*.
2. "8.3 % is a real improvement" → falsified: the same reward at 3M scores 0/60.
3. "A per-step stream on the success state necessarily harms learning" → falsified:
   probe D keeps the stream and scores 73.3 %, higher than the event-based probe E (65 %).
4. Terminal-form metric built on the *clipped total's* concentration → wrong (the clip
   flattens a +300 event to +20, the same size as a `+20` per-step term). Replaced by
   spike-vs-median analysis of the raw increment sequence.
5. "The mechanical gate has predictive power" → falsified (see P4).
6. "Adding time pressure would create the method's advantage" → falsified by inspecting
   the evidence channels (see P5).

---

## 8. Key artifacts

```
custom_envs/fragile_cargo_dock_env.py         the environment
envs/env_007/task_spec_anonymized.yaml        LLM-facing spec (original)
envs/env_007/task_spec_anonymized_v2.yaml     + exact dock geometry
prompts/eureka_01_initial_reward{,_v2..v5}.md initial-reward prompt revisions
prompts/eureka_02_reward_edit{,_v4}.md        edit-prompt revisions
configs/env007_fragilecargo_eureka.yaml            CREATE config
configs/env007_fragilecargo_eureka_baseline.yaml   EUREKA config
configs/env007_fragilecargo_eureka_v5.yaml         EUREKA with the v5/v4 prompts
pipeline/run_iterative_experiment.py          CREATE
pipeline/run_eureka_population.py             EUREKA baseline
analyze_terminal_dominance.py                 probe-based structural checker
diagnose_control.py                           per-episode failure-mechanism diagnosis
eval_fresh_seeds.py                           scoring on unseen seeds
pilot_generate_only.py                        generation-only A/B (16 candidates/arm)
measure_release_ballistics.py                 the lead-time measurement
runs/env_007/control_obs_only/reward.py       hand-written obs-only control
runs/env_007/ablation_probe/probe{A..E}*.py   single-variable ablations
runs/env_007/passthrough_probe/reward.py      native reward through the wrapper
runs/env_007/ABLATION_FINDINGS.md             the ablation write-up
runs/env_007/PILOT_TERMINAL_RULE.md           the pilot write-up
```

Reproduce the central numbers:

```
python measure_release_ballistics.py
python analyze_terminal_dominance.py --clip 20 --summary runs/env_007/ablation_probe/*.py
python eval_fresh_seeds.py --episodes 60 --seed-offset 30000 runs/env_007/ablation_probe/probeC
```
