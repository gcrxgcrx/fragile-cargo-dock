# SESSION STATE / RESUME POINT

> **Entry point for a fresh session: read `NEXT_SESSION.md` first.** It gives the first
> actions, the hard constraints and the pre-registered decision rule; this file is the
> detailed state behind it.
>
> Companion archives (both verbatim):
> `docs/external_reviews/chatgpt_answers_P1-P5.md`,
> `docs/external_reviews/deepseek_v41f_answers_P1-P5.md`.

Written 2026-09-20 for resuming after a context compaction. Supersedes the older
scattered notes for *resume* purposes; those remain as detailed write-ups.

---

## 0. Scope and standing constraints

* The user is **only running experiments** on `FragileCargoDock-v0`. The paper is not
  theirs. **All paper-reporting issues are out of scope** (they go to the paper's author).
  See `HANDOFF_QUESTIONS.md` section 6 for the list, so it is not re-litigated.
* **API keys**: `EUREKA_DEEPSEEK_API_KEY` for the EUREKA/GENERATION flow,
  `DEEPSEEK_API_KEY` for CREATE. The key must **never** be used for subagents or tasks
  outside those flows and must never be written to a file. It can be read at runtime from
  `runs/env_001/ablation_eureka_feedback_v4/reproduction/run_ablation_score_only_v4.ps1`
  (line 19) instead of being retyped.
* Generation only ever uses `deepseek-flash` (= DeepSeek-V4.1-Flash). `deepseek-v4.1f`
  returns HTTP 400. **Thinking mode must be off**: `DEEPSEEK_THINKING=disabled`.
* Interpreter: `D:\Code\python\research\llm_env_310\Scripts\python.exe`
  (bare `python` is msys64 Python 3.14 with no packages).
* Long runs must be `run_in_background: true` harness jobs, or detached `Start-Process`,
  otherwise they die with the spawning pwsh scope.
* Resource headroom: 32 logical cores, 31 GB RAM. 8 parallel trainings is fine.

---

## 1. The environment

`custom_envs/fragile_cargo_dock_env.py`, id `FragileCargoDock-v0`, authored by us.
Top-down Box2D, gravity 0, drag only. `obs` Box(19) clipped ±2, `action` Box(2),
400 steps, dt = 1/30.

* Success: crate fully inside the dock (`DOCK_HALF − CARGO_HALF = 0.42 − 0.30 = 0.12 m`
  on both axes), heading error < 30°, crate speed < 0.05 m/s, **held 10 consecutive
  steps**; the episode **terminates immediately** on success.
* Failure: cart or crate out of bounds, or ≥ 3 hard impacts (peak normal impulse > 5 N·s).
* Core mechanic: **the cart has no brake**; it must release early and let drag carry the
  crate in.

### Measured ballistics (`measure_release_ballistics.py`)

Coast distance = `(v0 − 0.05) / 2.0` (linear damping 2.0), dock at x = 2.6:

| release speed | coast | release at x | window | timing margin |
|---:|---:|---:|---:|---:|
| 0.2 | 0.074 | 2.526 | ±0.12 | 18.0 steps |
| 0.3 | 0.123 | 2.477 | ±0.12 | 12.0 steps |
| 0.5 | 0.222 | 2.378 | ±0.12 | 7.2 steps |
| 1.0 | 0.468 | 2.132 | ±0.12 | 3.6 steps |
| 2.0 | 0.959 | 1.641 | ±0.12 | 1.8 steps |

Settling takes 21–55 steps after release. So the task is a **ballistic commitment**, and
speed is the only thing that buys timing margin.

---

## 2. The harness

`training/reward_wrapper.py`: every **generated** reward goes through
`RewardOverrideWrapper`, which clips the per-step reward to `reward_clip`
(**default 20.0**, `training/train_sb3_wrapper.py:656`). The **native** reward is measured
with `reward_fn = None`, so it bypasses the wrapper and is never clipped.

Training config for everything below: `n_envs=6`, `gamma=0.999`,
`normalize_reward=true`, 3M steps per candidate unless stated.

---

## 3. Established measurements (60 **fresh** seeds 30000–30059)

### 3a. Harness is fine; the clip is not the problem

| policy | budget | fresh-60 |
|---|---:|---:|
| native reward, **bypassing** the wrapper | 3.0M | **96.8 %** |
| native reward, **through** the wrapper, clip 20 | 1.2M | **90.0 %** |
| native reward, through the wrapper, clip 600 | 1.2M | 80.0 % |

### 3b. The observation-only contract can express a solvable reward

All 1.2M, seed 0, clip 20. Each arm is the hand-written "control v1" with one change:

| arm | change vs control v1 | fresh-60 | Fisher vs 0/60 |
|---|---|---:|---:|
| control v1 | (incremental progress + approach + `+20`/step settled stream + boundary guard) | **0 %** | — |
| probe A | `+` true contact-impulse penalty (from `info`) | 40.0 % | <1e-4 |
| probe B | per-step stream → native one-off terminal event (`info`) | 45.0 % | <1e-4 |
| probe D | `+` **observation-only** closing-speed penalty | **73.3 %** | <1e-4 |
| probe E | obs gentleness **+** one-off event from module state + `obs[18]` (**no `info`**) | **65.0 %** | <1e-4 |
| probe C | true impulse **+** event form | **98.3 %** | <1e-4 |

The single decisive ingredient is a signal that penalises **closing speed**, not proximity.

### 3c. The LLM search has never produced a usable reward

| run | budget | best selection score | fresh-60 | success |
|---|---|---:|---:|---:|
| EUREKA v2 (old prompt/spec) | 10 × 3M | +8.093 | +7.027 | 0/60 |
| CREATE v2 (old prompt/spec) | 10 × 3M | +17.694 | +7.716 | 1/60 |
| pilot cand_03 (dock geometry + terminal rule) | 1.2M | +34.884 | +31.30 | **5/60 = 8.3 %** |
| same reward, full budget | 3.0M | +4.152 | +4.00 | 0/60 |

**Statistics (computed this session):** rule of three puts the 95 % upper bound of `0/60`
at **5 %**. `5/60 vs 0/60`: Fisher one-sided p = 0.0287, **two-sided p = 0.0573** — so the
8.3 % was *never* statistically distinguishable. All arms at 24/60 and above are <1e-4.

### 3d. Structural ruler over all rewards ever produced (no training)

`analyze_terminal_dominance.py --clip 20 --summary <rewards>`:

| contract | n | dock ÷ parked (single step) | episode-credit ratio |
|---|---:|---:|---:|
| old spec + old prompt | 20 | **1.02 – 1.24** | 0.025 – 0.031 |
| pilot (dock geometry + terminal rule) | 4 | **172 – 5123** | 0.289 – 1.278 |

Old twenty: the dock was worth 2–24 % more than not docking, and their largest dock
reward was exactly **20.00 = the clip**. The old search never tried to signal an event.

### 3e. The prompt ladder (dose-response) — **the key new result**

Three levels, generated this session, 16 candidates each, same context, T = 0.7,
**generation only, no training**:

* **L0 interface-only** = `prompts/eureka_01_initial_reward_v6_interface_only.md`
  (generic design prompt + harness facts; the incentive-conflict section removed)
* **L1 paper baseline** = `prompts/eureka_01_initial_reward.md` (= L0 + incentive-conflict
  section; this is what the paper's experiments used)
* **L2 oracle scaffold** = `prompts/eureka_01_initial_reward_v5.md`
  (L1 + gentleness formula + one-off-event skeleton + terminal dominance + boundary rule)

| check | **L0** | **L1** | **L2** |
|---|---:|---:|---:|
| writes a one-off terminal event | **1/16** | **0/16** | **16/16** |
| writes a gentleness term (gap > 0) | **0/16** | **2/16** | **15/16** |
| hover dominance ratio > 1 | 3/16 | 0/16 | 11/16 |
| **all four** | **0/16** | **0/16** | **11/16** |
| gentleness gap magnitude | — | 0.01–0.10 | **0.048–5.70** |

**Two external predictions confirmed**: removing the scaffold drops compliance from 16/16
to ~0 and raises inter-candidate variance. **Attribution**: L0 ≈ L1, so the leak is
**specifically the three rules I added**, not the paper's original prompt.

Raw data: `runs/env_007/prompt_ladder/{L0,L1,L2}/` (+ `manifest.json` per arm).

### 3f. Trajectory-ranking check — built, and it fails the same way

`trajectory_ranking_check.py` builds **physically reachable** trajectories from scripted
controllers (idle / push_forever / heuristic with release ∈ {0.05..0.80} / shove), takes
the environment's own return as the ordering oracle, and asks whether a candidate reward
orders whole trajectories correctly.

Measured (72 trajectories, 6 successes, 24 dock entries):

| reward | known fresh-60 | ranks success above failure | separation |
|---|---:|---:|---:|
| native passthrough | 90 % | 1.000 | 19.8 |
| probe E | 65.0 % | 1.000 | 20.4 |
| probe D | 73.3 % | 1.000 | 200.4 |
| probe A | 40.0 % | 1.000 | 199.6 |
| **control v1** | **0 %** | **1.000** | **199.6** |

**control v1 scores a perfect separation and still trains to 0 %.** So ranking correctness
on reachable trajectories is **not** sufficient for learnability — this is direct
empirical support for DeepSeek's argument that a reward is a *function* while the policy
is the *image after optimisation*, and it explains why every cheap probe failed.

(probe B/C show 0.528 only because the replay passes `info = {}`; they read
`info["official_reward_terms"]`. Cosmetic, does not affect the conclusion.)

### 3g. Decisive fact both external answers flagged and could not verify

**EUREKA's reward reflection already contains per-component activation rate.**
`pipeline/run_eureka_population.py`: `mean_eval_reward`, `mean_episode_length`,
`eval_episodes`, `termination_breakdown` (190–195); the training return curve (197–204);
a per-component `episode_sum_mean` table over training windows (242–244);
**a per-component `active_rate` table over training windows (245–248)**;
per-component `mean/abs_mean/min/max` (254–262). `magnitude_share` is just a
normalisation of `abs_mean` and is trivially reconstructable.

**Therefore C is a joint (channel-relative) property, not an environment property.** The
function's own docstring (178–180) claims the opposite ("no active-rate / share …") — the
docstring is **stale and contradicts the code**, and it misled both external models.

### 3h. Fidelity floor for any cheap-proxy ladder

Native reward (known-good) under these hyperparameters:

| steps | 0.1M | 0.2M | 0.3M | 0.5M | **0.6M** | 1.0M | 1.2M |
|---|---:|---:|---:|---:|---:|---:|---:|
| dock_rate | 0.00 | 0.00 | 0.00 | 0.00 | **0.20** | 0.80 | 0.90 |
| success | 0 | 0 | 0 | 0 | 0 | 0.50 | 0.80 |

Dock-related behaviour metrics are identically zero before ~0.6M, so a "3–10 % of budget"
rung cannot rank positive candidates. **The earliest informative rung is ~0.6M (20 %).**

---

## 4. Errors I made and falsified (do not re-litigate)

1. "The per-step clip is the blocker" → falsified: native through the wrapper at clip 20
   is 90 %; raising the clip to 600 made the best LLM candidate *worse*.
2. "8.3 % is a real improvement" → falsified: two-sided p = 0.0573, and the same reward at
   3M scores 0/60.
3. "A per-step settled stream necessarily harms learning" → falsified: probe D keeps the
   stream and scores 73.3 % > probe E's 65.0 %.
4. Terminal-form metric built on the *clipped total's* concentration → wrong (the clip
   flattens +300 to +20, the same size as a +20/step term). Replaced by spike-vs-median.
5. "The mechanical gate has predictive power" → falsified: 15/16 gate-passers, three
   trained, all 0/60.
6. "Initial-reward mass near the threshold predicts the method gap" → wrong; it predicts
   the benefit of best-of-k sampling and can point the other way.
7. "probe D proves A holds (localized edit)" → conflates edit distance with search
   reachability; it is a human-oracle artifact.
8. "Time pressure would create the method's advantage" → falsified: `mean_episode_length`
   and `termination_breakdown` are in **both** evidence channels.
9. "The shared scaffold is the main reason there is no method difference" → unproven; the
   data only shows *structural* convergence.
10. "~1000× dynamic range" as an argument for the gate → invalid; monotone transforms
    inflate dynamic range without changing ordering.

---

## 5. RESULT of the outcome-variance test (done — decision rule fired)

**Setup.** Eight detached 1.2M-step trainings on candidates chosen to span the structural
checks: `L2/{00,05}` (all four checks pass), `L2/{04,14}` (event but not the others),
`L0/{02,11,13,00}` (almost nothing). Joined with the four structural checks by
`ladder_analysis.py`; scored on the held-out seeds 30000–30059.

**Result: every one of the eight is 0/60.**

| candidate | event | 2nd | term ratio | gentle gap | hover ratio | traj succ>fail | fresh |
|---|---:|---:|---:|---:|---:|---:|---:|
| L2/cand_00 | 1 | 1 | inf | 1.140 | inf | 0.773 | **0/60** |
| L2/cand_05 | 1 | 1 | inf | 5.700 | inf | 0.768 | **0/60** |
| L2/cand_04 | 1 | 1 | inf | 1.900 | 0.25 | 0.654 | **0/60** |
| L2/cand_14 | 1 | 1 | inf | −4.325 | 0.07 | **1.000** | **0/60** |
| L0/cand_02 | 1 | 1 | inf | 0.000 | inf | 0.268 | **0/60** |
| L0/cand_11 | 0 | 0 | 0.00 | 0.000 | 0.03 | 0.659 | **0/60** |
| L0/cand_13 | 0 | 0 | 0.00 | 0.000 | 0.05 | 0.841 | **0/60** |
| L0/cand_00 | 0 | 0 | 0.00 | 0.000 | inf | 0.649 | **0/60** |

Outcome variance is still zero, so the correlations are formally undefined — but the
substantive reading is stronger than "undefined": **the entire range of every structural
check maps onto the same outcome.** Mean return does vary (−67.3 … +1.55), so the runs are
not degenerate; only success is uniformly absent.

Note also that `L2/cand_00` and `L2/cand_05` are the **strengthened** gentleness variants
(gap 1.14 and 5.70, against the 0.048 that every v4 candidate copied), and they fail too.
With the three earlier v4 candidates this makes **11 scaffold-compliant LLM rewards, all
0/60.**

### Decision rule — which branch fired

> "If every check is zero-correlation → cheap probes are closed for selection."

**This branch fired.** Cheap structural probes cannot be used as a positive selector on
this environment. The measured fallback is selection by short training at the **~0.6M**
fidelity floor (see §3h) with more seeds, keeping 30000–30059 permanently held out.

### 5.1 Follow-up: that 0.6M fallback was measured, and it does **not** work

Added 06:13, same session, after running the fallback rather than assuming it. Full write-up
and raw data: `runs/env_007/LADDER_FINDINGS.md` §5,
`runs/env_007/LADDER_DECISION_PREREGISTRATION.md` §3b–3c (protocol fixed in advance),
`runs/env_007/rung_06m/`.

Design: the six hand-written arms (whose 1.2M outcomes span 0 % → 98.3 %) retrained at
**0.6M** with seeds {0,1,2} = 18 runs, plus the eight ladder candidates at 0.6M (8 runs);
all scored on a **new** fresh block, seeds **32000–32059**, so 30000–30059 stays pristine.

**Result — the 0.6M rung inverts the ranking.** `rho(0.6M success, 1.2M success) = −0.486`
(bar was ≥ +0.60), and the mean within-candidate seed sd (0.236) exceeds the
between-candidate sd (0.149). The rung ranks `control_v1` — 0 % at 1.2M — **second best of
six**, and puts `probeC` (98.3 % at 1.2M) fourth. A search selecting on 0.6M would have
preferentially kept the reward that reward-hacks. More seeds cannot fix this: the means are
already 3-seed means, so extra seeds shrink variance, not the bias, and the same means give
rho ≈ −0.49.

`rho(0.6M dock_rate, 1.2M dock_rate) = +0.543` sharpens the reason: the *docking behaviour*
is partly preserved at 0.6M while the *success* is not, because at 0.6M every arm docks a
lot (0.46–0.85) and few satisfy the strict predicate. "The crate reached the dock" — the
obvious cheap proxy — is the near-miss the predicate exists to exclude.

**Why the ranking inverts: training longer destroys the behaviour.** Comparing seed 0 across
budgets, `control_v1` goes **39/60 at 0.6M → 0/60 at 1.2M**, while all five hand-designed
probes *improve* (A 12→26, B 7→22, C 25→59, D 0→44, E 0→46). `control_v1` is the arm whose
reward is dominated by a farmable dense term, and its 0.6M policy does dock — further PPO
trades that for the unconditionally available progress/approach reward. This is reward
hacking emerging with optimisation, and it is the same phenomenon as §3c's 5/60 at 1.2M
collapsing to 0/60 at 3M.

**A second, independent readout was also rejected.** The 1.2M component tables show every
working arm carrying ~96 % of its reward mass in a positive *sparse* term and every failing
arm carrying none (`control v1` and `L0/cand_{02,11,13}` only have dense farming terms,
`L0/cand_00`/`L2/cand_00` mass in sparse *penalties*, `L2/cand_05` nothing at all). Frozen
as a readout and tested at 0.6M on 14 candidates: precision **1.000**, recall **0.400**,
AUPRC **0.877** (baseline 0.357) → rejected by the pre-registered bar (recall ≥ 0.60). It
never certified a bad arm but missed 3 of 5 good ones, because at 0.6M the terminal term has
not yet taken over. The same readout separates 14/14 at **1.2M** — the budget of just
training the candidate and measuring it.

**Consequence — the cheap-selector programme is closed from the selection side.** No
training-free or short-training instrument tested in this session separates these rewards:
structural checks (no correlation), scripted-trajectory ranking (perfect separation, useless
— §3f), realised component activity at 0.6M (recall 0.40), and only at 1.2M does anything
work. The LLM pool is dead at both rungs: all eight ladder candidates at 0.6M are **0/60
with `dock_entered = 0.00`**, so no selector could have rescued a pool with no positive
member. Selection cannot be made cheaper than evaluation here, which points the remaining
work at the two generation-side hypotheses below (H1/H2), not at better selectors.

### 5.2 Intermediate rung (1.0M): pre-registered, and it fails too

Added 07:05. The 0.6M failure left one rung open — is the readout merely *early*? Protocol
and bar were fixed before the runs (`LADDER_DECISION_PREREGISTRATION.md` §3d). 18 runs
(six arms × seeds {0,1,2}) at 1.0M, `runs/env_007/rung_10m/`.

* **Outcome-based:** `rho(1.0M, 1.2M success) = +0.371` (bar ≥ +0.60) → REJECT. It is now
  positive rather than inverted (−0.486 at 0.6M) and the noise condition nearly passes
  (within 0.252 ≈ between 0.289), so the rung improves with budget without arriving.
  `control_v1` (0 % at 1.2M) is still ranked **4th of 6**.
* **Readout:** precision 0.833, **recall 1.000**, AUPRC 0.877 → REJECT on the pre-registered
  condition that `control_v1` must not be certified. It *is* certified, in **3/3** seeds.
  So the 0.6M diagnosis was right about sensitivity (0.40 → 1.00 once the terminal term
  stabilises) and the error merely changes type: at 1.0M `control_v1`'s settled **stream**
  (share 0.971, +238/episode) is indistinguishable from the probes' settled **event**. At
  1.2M the same arm is rejected only because its policy has by then abandoned docking —
  **the readout's 14/14 at 1.2M is partly an accident of that degeneration.**

Three rungs, two instruments, no usable selector below the reference budget: **the cheapest
rung at which any selector works is the budget of the thing being selected.** Caveat worth
carrying forward: the 1.2M reference is a project convention, not ground truth — §3c shows
1.2M → 3M can flip an outcome (the pilot's 5/60 → 0/60), and if the target itself moves with
budget then no fixed rung can be validated against it.

### The bigger conclusion this forces

Across the whole session the LLM has produced roughly **35 reward functions** (24 in the
original CREATE/EUREKA runs, 3 v4, 8 here) and **not one delivers the crate**, while
hand-written rewards on the same contract reach **40 % / 45 % / 65 % / 73 % / 98 %**.
So on this environment the bottleneck is not diagnosis, not the harness, and not the
criteria — **the generator itself never emits a reward whose global landscape PPO can
follow.** Component presence and local magnitude checks do not determine that landscape.

Two hypotheses remain untested and should be the next thing on the table:

1. **Generator strength.** The paper's env_001/env_002 runs used `deepseek-v4-pro`; every
   env_007 run here used `deepseek-flash` (= DeepSeek-V4.1-Flash). The user holds that
   V4.1F is the stronger model, but this was never tested on this task. Re-running the
   *same* L2 prompt with a stronger generator is a cheap, decisive check of whether the
   ceiling is the generator's or the contract's.
2. **Search, not generation.** If a stronger generator still fails, then the reward the
   task needs is not reachable by one-shot generation at all, which is exactly the regime
   where a *repair* loop should matter — and where the trajectory-evidence extension
   (section 5 below) becomes the thing to build.

### Standing conclusion from the two external answers (adjudicated, both accepted)

* The correct object for "evidence asymmetry" is **EVSI**, not an environment property;
  C is `C(E, O, X, B)` — a joint property of environment, evidence channel, edit
  operator class and budget.
* Fixing the evidence channel **is** a legitimate method contribution, but only if the
  channel is mounted on one side only and the asymmetry is reported explicitly.
* The remaining theoretical difficulty: C needs an edit that is **small enough to count as
  localized** yet **large enough that blind search cannot hit it**. Measured hint: the
  gentleness weight needed to stop being swamped is 30–100× the natural value.

---

### 5.3 The LLM failure is two-layer, and a two-edit repair reaches 38.3 % (added 20:45)

Full write-up: `runs/env_007/REPAIR_TEST_FINDINGS.md`; protocols and predictions fixed in
advance in `REPAIR_TEST_PREREGISTRATION.md`, `ALIGN_FIX_PREREGISTRATION.md`,
`RECIPE_REPLICATION_PREREGISTRATION.md`; diagnostics in `ADVANTAGE_PROBE_FINDINGS.md`.

**Headline.** `L2/cand_00` and `L2/cand_14` — two independent LLM generations, both 0/60 —
each become **23/60 = 38.3 %** (dock rate 0.98) after the same two localized edits, on fresh
seeds 32000–32059. That is the first time an LLM-derived reward in this project has ever
delivered the crate: one-sided Fisher **p = 8.7e-09** against 0/60, and above the previous
best LLM result ever recorded (5/60 = 8.3 %, itself p = 0.0573 two-sided). 38.3 % sits
between hand-written `probeB` (36.7 %) and `probeA` (43.3 %).

**The recipe (two edits, neither sufficient alone).**

1. multiply the candidate's **own approach term** by ~50 (`crate_to_dock_progress`
   12 → 600 for `cand_00`, 6 → 300 for `cand_14`): makes *acting* profitable;
   measured effect alone (`r04`): dock entry 0.00 → **0.97**, goal distance 4.148 m → 0.140 m,
   still 0/60 because nothing pays for settling;
2. pay a **dense +20/step payoff on the instantaneous success predicate** (inside ∧ aligned ∧
   slow). Alone it does *nothing*: `r06`, `r07`, `g01_stream` are **bit-for-bit equivalent to
   their copy controls** (identical reward, identical policy, identical 0/60), because the
   stream is gated on a predicate the policy never reaches.

**Mechanism, confirmed.** The candidate's own one-off +300 payoff was always correctly
written and never fired; after the repair `success_event` activation goes 0.0000 → 0.0012 /
0.0017 and contributes 12–29 % of the realised reward mass, with 9–11 of 20 in-training
episodes terminating.

**Controls that matter.** Adding `probeD`'s gentleness **hurts** this candidate (`r08`:
0/60, dock 0.00; `r10`: 0/60, dock 0.27) — the term that is decisive in the hand-written
family is harmful when stacked on the candidate's own `soft_contact`/`dock_speed_penalty`.
Raising its own speed penalty ×100 (`r11`) is also 0/60 (return −1.65, worse than base) —
predicted *before* its result was read by the advantage probe (`A` = −12.8 vs −0.26).

**Why they fail at all: a sign problem, not a structural one.** On the scripted library the
base reward already orders correctly — success +56, pushing +6.4, idle −0.8 per episode — yet
the trained policy sits at ≈ −0.07 and never moves the crate. The **per-step advantage of
acting over idling** is only ≈0.018; the advantage-scale probe measures the whole reward at
**A = −0.259/step**, i.e. *acting is worse than doing nothing*, so the "failure to learn" is
in fact correct optimisation of a mis-scaled reward. The ×50 edit moves A to **+0.113** and
the cart moves. So learnability needs **both** `A > 0` **and** the reward's argmax over
reachable behaviour being the success behaviour; every cheap instrument tried measures at
most the first (`ADVANTAGE_PROBE_FINDINGS.md` has the full table).

**A second §3f instance, with the root cause found and fixed.** `L0/cand_02`'s alignment term
is **inverted**: `heading_align = 1 − |obs[10]|` while `DOCK_ANGLE = 0` and success needs
`|cargo_angle| < 30°`, so its 94 %-of-mass `crate_docking_quality` term is maximised by
holding the crate perpendicular — a state that can never succeed, i.e. reward hacking with a
one-token root cause. Fixing that one subscript moves the training-free ordering check from
**0.038 → 0.702**, and the arm **still trains to 0/60** (dock 0.00), because its per-step
advantage stays at `A ≈ −0.0004`. Ranking repaired, learnability unchanged.

**Caveats, all load-bearing.** Every arm here is **oracle-authored** — this measures that a
repairable target *exists* and how large it is (two edits), never that EUREKA/CREATE would
find it. The recipe was derived on the L2 family and replicated on two of its members; the
**L0 family fails differently** (its dense terms pay for the wrong behaviour) and its repair
is untested. `LADDER_FINDINGS.md` §5 stands: no cheap selector works below the full budget.

---

### 5.4 Does the evidence channel that already exists enable repair? **No** (added 23:35)

Write-up: `runs/env_007/REPAIR_LOOP_FINDINGS.md`; protocol, sham construction, N, bars and the
two-stage rule fixed in `REPAIR_LOOP_PREREGISTRATION.md` before anything was generated.
Driver `run_repair_loop.py` (the pipeline's own operator + report, not a new channel).

**Design.** Eight targets (the 4 L2 + 4 L0 ladder candidates), one repair each per arm.
Real arm = `build_reward_reflection(<target's 1.2M summary>)`; sham arm = the byte-identical
report with every per-component table's rows shuffled and each value column shuffled
independently (verified: 3 tables per report, everything else identical). Operator, prompt,
model (`deepseek-flash`, T = 0.7), context and N identical. 16 repairs generated, all valid.

The real channel **does** contain the decisive evidence, verified before generating:
`L2_cand_00` shows `success_event` and `enter_dock_event` at **0.0 % in every window**, and
`L0_cand_02` shows `crate_docking_quality` at **100 % from the first window**.

**Result (fresh seeds 33000–33059, a block used for the first time):**

| arm | hits | detail |
|---|---:|---|
| real | **0/8** | all 0/60, `dock_entered` **0.00** for every one |
| sham | **1/8** | `sham/L2_cand_14` = 5/60 = 8.3 % (dock 0.55) |

one-sided Fisher (real > sham) = **1.0000**. **E1, E2 and E3 all fail**, and E4 is
unsupported (`A > 0` in 3/8 real repairs with 0 successes, 7/8 sham with the 1 success).
My prior — that the real table would beat the sham because it contains both signals — was
wrong and is recorded as such.

**Consequences.**

* The negative result on this environment is **capability-bounded, not evidence-bounded**:
  given the component-activation evidence it already has, the operator does not produce the
  repair. The pre-registered consequence is that **adding channels — including P5's
  trajectory-evidence extension — is predicted to be worthless here**.
* The one hit is in the *sham* arm and equals the project's historical luck magnitude
  (5/60 = 8.3 %, the old best-ever LLM candidate, which collapsed at full budget and was never
  significant). The sham repairs also carried the largest per-step advantages of the
  experiment (up to +6.6), i.e. dense farmable terms — a known failure mode.
* The gap is large and measured: the oracle two-edit repair hits **2/2 targets** (23/60 each,
  dock 0.97) while the model's own edits hit **0/8 with the real evidence**. Rule of three
  bounds the real arm's hit rate at ≤ 37.5 % (95 %); N = 8 cannot exclude a small effect, but
  it excludes the large effect the hypothesis needed.
* Not ruled out, and the natural next measurement: **iteration** (repair → retrain → reflect)
  or a stronger operator. This tested one round with one operator.

---

### 5.5 The v7 prompt: removing a self-imposed prohibition (added 01:50)

Pre-registration and full write-up: `runs/env_007/V7_PROMPT_PREREGISTRATION.md`; prompt
`prompts/eureka_01_initial_reward_v7.md` (v5 untouched, diff in `v7_prompt.diff`).

**The finding that motivated it.** The scaffold in use **forbade the ingredient the measured
repair needs**: v5 line 112 — "完成奖励必须是一次性事件，不能在'成功状态'上每步给分（违反即无效）"
— enforced by self-check ④, which required the model to *rewrite* any per-step settled payoff.
The two-edit repair that takes two candidates to 23/60 needs exactly that form. And the ban's
justification was a mis-attributed measurement: the grinding it cites is `control_v1`
(`A = −0.26`); the cause was the weak approach term, not the stream. v5's own prescribed cure
for "gets in but cannot stop" (a one-off dock-entry bonus) is measured not to work (`r04`:
dock 0.97, 0/60).

**v7** replaces the ban with a conditional requirement (one-off event **and** per-step settled
payoff), mandates the settled payoff, inverts self-check ④, adds self-check ⑤ (the advantage
calibration: `R_push > R_idle` by at least the largest penalty's per-step magnitude,
`R_settled > R_push`, no thousand-fold compression of the positive terms), and forbids stacking
gentleness duplicates.

**Result** (8 candidates, fresh seeds 34000–34059, 60 episodes):

* mechanical effect complete: **8/8 have the per-step settled payoff** (v5 family 0/2);
* **`cand_01` reaches `dock_entered` 0.18** — the first *unedited* LLM candidate in this
  project ever to enter the dock (all 35+ previous, v1–v6 plus the 16 channel repairs,
  measured 0.00);
* **1/8 hits (3/60 = 5.0 %)**, one-sided Fisher vs the 0/16 v5-family baseline = **0.3333** →
  not distinguishable from noise; outside the pre-declared stage-2 band, so the experiment
  stopped at one stage (V7-1 and V7-4 hold nominally);
* in-training returns were the best ever for this generator (+3.05/+3.30), and within v7 the
  in-training return picked the docking candidate — unlike v5, where the best in-training
  return was the reward hacker.

**Reading:** the prompt fix is real but insufficient. 5 % is *below* the historical best-ever
LLM observation (5/60 = 8.3 %, non-significant) and 7.7× below the oracle repair (38.3 %), and
7 of 8 candidates still never dock — exactly the four with `A < 0`. The cheap next lever the
data points at: make the advantage calibration **mechanical** in the pipeline's existing
`validate_code` path (reject `A ≤ 0`, or non-monotone `R_idle/R_push/R_settled`), which checks
the two *measured* necessary conditions rather than the *structural* gate that failed before
(§4 error #5).

---

### 5.6 The working reward is a narrow ridge: nine perturbations, all 0/60 (added 02:20)

> **Superseded in part — read §5.8 before quoting this section.** The seed-robustness runs showed
> that the "all 0/60" reading conflates two things: arms whose `dock_entered` **also collapsed**
> (the overshoot cliffs, proximity, funnel) are real behavioural changes, while arms separated only
> by a small *success* difference are inside the settling variance. §5.8 has the corrected table
> and the two-factor decomposition (reachability vs settling).

Pre-registration and full tables: `runs/env_007/OVERSHOOT_ABLATION_PREREGISTRATION.md` and
`APPROACH_ABLATION_PREREGISTRATION.md` (both written before their runs; the advantage probe's
predictions are in them and two of the three were confirmed blind).

Two domain-expert hypotheses were tested as single-axis edits on the **known-working** reward
(`r09`, the two-edit repair of `L2/cand_00`), each with a **byte-identical copy of that reward as
a matched baseline in the same fresh block 35000–35059** — so no cross-block comparison is ever
used (block drift is ±12 points):

| the idea | arm | fresh-60 | dock | baseline in the same block |
|---|---|---:|---:|---|
| **make overshoot very negative** | `o01_overshoot` (−20/step cliff) | **0/60** | 0.00 | **16/60, dock 0.90** |
| same, 10× harder | `o04_overshoot_x10` (−200/step) | **0/60** | 0.00 | " |
| add a gentleness/closing-speed penalty | `o02_gentle` | **1/60** | 0.35 | " |
| overshoot cliff + gentleness | `o03_both` | **0/60** | 0.00 | " |
| **reward approaching, state form** | `p01_proximity` | **0/60** | 0.02 | " |
| approaching, delta form ×4 stronger | `p02_progress_x200` | **0/60** | 0.72 | " |
| **approaching, "near AND slow" funnel** | `p05_funnel` | **0/60** | 0.07 | " |
| proximity + gentleness / ×200 + gentleness | `p04`, `p03` | **0/60** | 0.10 / 0.53 | " |

**Every one of the nine perturbations falls to 0/60** (the earlier repair-study arms `r01`–`r08`,
`r10`, `r11` are the same story). Only the recipe itself delivers: **23/60 on block 32000–32059 and
16/60 on block 35000–35059**. So the working point is a **narrow ridge** — a specific coefficient
(~×50 on the candidate's own delta-progress term) plus a specific payoff form (a dense per-step
payoff on the *instantaneous success predicate*). Perturbing either axis up or down, adding or
removing, falls off it. Two failure modes are visible in `dock_entered`: **too much pull** (×200)
arrives but cannot settle (dock 0.72); **dense near/reward-shaped terms** hover without entering
(dock 0.02–0.10).

**Why the overshoot cliff destroys a working reward** (measured, not interpreted): with no brake
the only route to the dock is to push and let drag stop the crate, so arriving trajectories pass
near the far edge; penalising that at −20/step makes the **approach itself** unprofitable, `A`
flips from +0.141 to −0.253, and the optimum becomes *not approaching* (dock 0.90 → 0.00). Same
signature as `r11` and the whole v5/v7 immobile family.

**Gentleness, replicated three times**: it is decisive only *relative to a reward that lacks it*
(probe D vs control v1). Stacked on a reward that already penalises closing speed
(`r09` has `soft_contact = −1.2·contact·closing`) it suppresses the push: `r08` 0/60 dock 0.00,
`r10` 0/60 dock 0.27, `o02` 1/60 dock 0.35 against a 16/60 baseline.

**Limits, stated:** one seed per arm (a 0/60 arm is bounded at ≤5 % by the rule of three, so it
means "badly degraded"); the variants are single-point samples, so the claim is "the basin is
narrow and one-axis edits fall out of it", **not** "the optimum is exactly ×50" (×75/×100 and other
funnel decay constants are untested). Untested mechanism hypothesis for why a *positive* dense
addition (p05) can also break delivery: VecNormalize's reward clipping distorts the effective
per-step weighting of all terms once a high-duty-cycle term raises the raw scale.

**Consequence for the prompt.** "Reward approaching" is not a usable *principle* — all four forms
fail. A v8 prompt must therefore specify the recipe **quantitatively** (coefficient scale + dense
payoff on the instantaneous predicate) and be paired with the mechanical gate (`A > 0`, 2×2
ordering). And the methodological consequence: if the form must be spelled out this tightly, the
search's role collapses to **parameter tuning inside a specified form** — which is measurable, and
which is what the operator ablation should test (seed the loop off-ridge; can its diagnosis
recover a known-good parameter?).

---

### 5.7 The scale axis is JAGGED, not a smooth ridge — and the probe cannot see it (added 02:50)

**Read `runs/env_007/RIDGE_WIDTH_PREREGISTRATION.md` (which carries an explicit note that this one
was NOT pre-registered) before quoting §5.6's "narrow ridge" phrasing.** A single-axis sweep of the
one coefficient that the working recipe sets (the candidate's own delta-progress term, baseline 600
= ×50 of the original 12), all on the same fresh block 35000–35059:

| coefficient | ×original | fresh-60 | `dock_entered` | terminations (in-training) |
|---|---:|---:|---:|---:|
| 600 | ×50 | 16/60 | 0.90 | 9/20 |
| **900** | ×75 | **0/60** | **0.07** | **0/20** |
| **1200** | ×100 | **31/60 = 51.7 %** | **0.95** | 9/20 |
| 2400 | ×200 | 0/60 | 0.72 | — |

`q02` at **31/60 = 51.7 % is the best result of the session** (31 vs 16 for the matched baseline in
the same block, Fisher one-sided p ≈ 0.004; better than the previous best LLM-derived candidate
5/60 by 6×, and above hand-written `probeA` 43.3 %). But the *shape* is the finding: **×50 works,
×75 collapses to 0/60, ×100 works better than either, ×200 collapses again** — a jagged success
surface in a single scalar, not a smooth band. The in-training signatures agree the arms trained to
different policies (×75: 0/20 terminations, `success_event` active 0.0000; ×100: 9/20, active
0.0012), so this is behavioural, not eval wobble.

**The instrument is blind to it.** `advantage_scale_probe.py`: `A` = +0.141 (×50), +0.331 (×75),
+0.521 (×100), **+1.281 (×200)** — smoothly monotone in the coefficient, while the outcomes are
jagged, so it ranks the worst arm highest. **Consequence: the planned mechanical gate built on
`A > 0` is insufficient — it would pass ×75 and ×200, both 0/60.** `A` can only certify a
*necessary* condition. This is the second independent negative for the probe as a selector.

**Caveat and the measurement in flight:** one seed per point (the same reward file has scored 23/60
and 16/60 on two blocks, so run-to-run variation exists). Seeds 1 and 2 for both ×75 and ×100 are
training (`runs/env_007/ridge_seeds_spec.json`). If ×75 stays ≈0 across three seeds → the surface is
genuinely rugged, local/gradient calibration is misled, and sample-and-train is the only strategy;
if ×75 spreads widely → it is a *noisy success probability*, and **every n = 1 comparison in §§5.1–5.6
needs that caveat attached** (including the ablation drops).

**This flips the earlier method prediction.** With a rugged scalar axis, a diagnosis-and-nudge loop
(CREATE) is actively misled — it sees ×50 work, nudges to ×75, sees failure, and backs off, while
×100 was better. A **sampler** (EUREKA's 4 draws per generation) is better suited. So under
ruggedness **breadth beats depth**, the opposite of the reasoning given before this measurement.

---

### 5.8 Seed robustness: the difficulty is SETTLING, and it is near-coin-flip (added 03:00)

§5.7's single points were followed by seeds 1 and 2 for both ×75 and ×100 (same block 35000–35059,
same 1.2M protocol). **Resolve any conflict in favour of this table.**

| coefficient | seed 0 | seed 1 | seed 2 | mean success | `dock_entered` per seed |
|---|---:|---:|---:|---:|---|
| ×50 (600) | 16/60 | — | — | (16/60) | 0.90 |
| **×75 (900)** | 0/60 | 4/60 | 0/60 | **2.2 %** | 0.07 / 0.48 / 0.00 |
| **×100 (1200)** | 31/60 | **43/60** | 0/60 | **41 %** | **0.95 / 0.78 / 0.95** |

Read the failure structure, not just the success column:

* **Reachability is set fairly reliably by the coefficient.** ×100 docks in **0.95 / 0.78 / 0.95**
  of episodes across three seeds; ×75 is erratic (0.07 / 0.48 / 0.00). "Does the policy get the
  crate to the dock" is close to a function of the coefficient.
* **Settling is the high-variance, binding step.** ×100 seed 2 is the decisive datum: **dock 0.95
  with 0/60** — it delivers every episode and never completes. ×100's successes are {31, 43, **0**};
  ×75's are {0, 4, 0}. So "success" is roughly `P(reach) × P(settle)`, and the second factor is
  near-coin-flip at fixed settings.

**Corrected description of the "jagged surface":** not pipeline chaos. Two attractors —
(A) *farms the dense progress term without entering* (×75's dominant component is
`crate_to_dock_progress` at 79 %, `enter_dock_event` 0 %, dock 0.07) and (B) *delivers and settles*
(×100: `settled_stream` 40 % / `progress` 37 %, dock 0.95) — with a **coefficient-dependent, plainly
non-monotone probability of landing in B**, plus heavy per-seed variance in the settling step.
×75 is genuinely bad (mean 2.2 % over three seeds), ×100 is genuinely good (mean 41 %) and ×50 is
intermediate (16/60 in one seed).

**But the headline positive result improves:** ×100 seed 1 = **43/60 = 71.7 %** is the best number
this project has produced (previous best LLM-derived candidate 5/60 = 8.3 %; hand-written `probeA`
43.3 %). It must be reported as *one seed of three* with the mean 41 % and the dead seed beside it.

### Artifact checks that were run before believing any of it (`diagnose_ridge.py`)

1. **The edit is exactly the coefficient.** Diffing each variant against the working reward leaves
   only the header comments and the one number (600 → 900 / 1200 / 2400).
2. **No silent fallback or config drift.** All four arms: `n_envs=6`, `reward_clip=20.0`,
   `total_timesteps=1200000`, `error_fallback=zero`, `reward_source=generated_function`,
   **`reward_error_count_max=0`**, `final_policy_component_error_count=0`, eval 20 episodes at
   `seed_offset=10000`.
3. **Not an evaluation artefact.** The in-training self-evaluation (20 episodes, seeds 10000–10019)
   agrees in sign with the fresh 60-episode block in every arm (×50 9/20 terminated; ×75 0/20;
   ×100 9/20; ×200 2/20).
4. **The policies really differ.** Different dominant components, and different training curves:
   ×75's mean generated reward rises to 0.863 then **falls to 0.575**, while ×100 climbs
   monotonically to 1.807.

### Qualification of §5.6's "nine perturbations, all 0/60"

That claim must now be split:

* **Conclusions that survive** — the arms whose **`dock_entered` collapsed as well** (the overshoot
  cliff `o01`/`o03`/`o04` at 0.00, proximity `p01` 0.02, funnel `p05` 0.07): the *behaviour* changed
  (avoidance / hovering), which settling variance cannot explain. The mechanism stands: a positional
  cliff punishes the only available route, `A` flips negative and the optimum becomes not approaching.
* **Conclusions that must be weakened** — anything resting only on a small difference in *success*
  between single-seed arms. At a fixed setting the settling step is near-coin-flip, so a one-seed
  candidate comparison is one lottery draw.

### Methodological consequence (goes to the paper's tables, not just this repo)

**Single-seed candidate comparisons are unreliable here.** Any "0/60 vs 1/60"-style contrast is
inside the settling variance. Report **`dock_entered` (stable) and success (high variance)
separately**, and state the seed count. `configs/env007_fragilecargo_eureka.yaml` has
`multi_seed.num_seeds: 3` enabled, but which seed count produced the paper's env_007 tables has
**not been verified** — check it before quoting any per-candidate number.

### Decision recorded (user, 03:00)

* **No further multi-seed confirmation runs.** The variance is characterised well enough to act on.
* **The coefficient is deliberately left to the LLM/search, not fixed by us.** The next prompt
  revision (v8) must prescribe the **form** (the two edits) and leave the numeric scale to the
  generator — with the measured caveat that scale is a first-order, non-monotone variable here, so
  the *sampling* loop (not a gradient-following one) is the appropriate search over it.

---

## 6. File inventory (new this session)

```
HANDOFF_QUESTIONS.md              scope split + P1–P5 (the ask) + out-of-scope paper issues
SESSION_STATE.md                  this file
analyze_terminal_dominance.py     probe-based structural checker (4 checks)
trajectory_ranking_check.py       reachable-trajectory ordering check (+ --describe)
diagnose_control.py               per-episode failure-mechanism diagnosis
eval_fresh_seeds.py               scoring on unseen seeds
pilot_generate_only.py            generation-only A/B (N candidates per prompt)
pilot_train_existing.py           train existing reward files (clip A/B)
pilot_ab_summary.py               clip A/B summariser
measure_release_ballistics.py     the lead-time law
ladder_analysis.py                joins checks with outcomes, per-check Spearman
configs/env007_fragilecargo_eureka.yaml             CREATE
configs/env007_fragilecargo_eureka_baseline.yaml    EUREKA
configs/env007_fragilecargo_eureka_v5.yaml          EUREKA with v5/v4 prompts
configs/env007_terminal_rule_pilot.yaml             pilot context config
configs/env007_terminal_rule_pilot_clip600.yaml     clip A/B
prompts/eureka_01_initial_reward.md                 L1 (paper baseline)
prompts/eureka_01_initial_reward_v2..v5.md          successive scaffolds
prompts/eureka_01_initial_reward_v6_interface_only.md   L0
prompts/eureka_02_reward_edit{,_v4}.md              edit prompts
envs/env_007/task_spec_anonymized{,_v2}.yaml        spec, and spec + dock geometry
runs/env_007/control_obs_only/reward.py             hand-written obs-only control
runs/env_007/ablation_probe/probe{A..E}*.py         single-variable ablations
runs/env_007/passthrough_probe/reward.py            native reward through the wrapper
runs/env_007/prompt_ladder/{L0,L1,L2}/               the dose-response arms
runs/env_007/ladder_train/                          the 8 outcome-variance trainings
runs/env_007/rung_06m/                               Branch B: 18 x 0.6M (6 arms x 3 seeds) + 8 ladder, evals, readout
runs/env_007/rung_10m/                               intermediate rung: 18 x 1.0M + eval + readout
runs/env_007/repair_test/                            14 oracle repair arms for L2/cand_00 (+ evals, component tables)
runs/env_007/align_fix_test/                         the inverted-alignment root-cause experiment on L0/cand_02
runs/env_007/recipe_replication/                     the two-edit recipe replicated on L2/cand_14 (23/60)
runs/env_007/REPAIR_TEST_FINDINGS.md                 **the repair result: 0/60 -> 23/60 on two candidates**
runs/env_007/{REPAIR_TEST,ALIGN_FIX,RECIPE_REPLICATION}_PREREGISTRATION.md   protocols + predictions, fixed before runs
runs/env_007/ADVANTAGE_PROBE_FINDINGS.md             the advantage-scale diagnostic (A<=0 => inaction is optimal)
runs/env_007/overshoot_ablation/                     overshoot cliff / gentleness ablation (5 arms) + eval
runs/env_007/approach_ablation/                      proximity / delta / funnel ablation (5 arms) + eval
runs/env_007/ridge_width/                            coefficient sweep x75/x100 + seed robustness (6 runs) + evals
runs/env_007/COMPUTE_LEDGER.md                       measured throughput + cost of every planned step
runs/env_007/OVERSHOOT_ABLATION_PREREGISTRATION.md   the overshoot cliff: predictions + outcome
runs/env_007/APPROACH_ABLATION_PREREGISTRATION.md    proximity/delta/funnel: predictions + outcome
runs/env_007/RIDGE_WIDTH_PREREGISTRATION.md          coefficient sweep + seed table (NOT pre-registered; banner says so)
runs/env_007/RIDGE_RECOVERY_PREREGISTRATION.md       the next experiment: can a loop's diagnosis recover the ridge?
runs/env_007/repair_loop/                            16 evidence-channel repairs (8 real + 8 sham) + evals
runs/env_007/REPAIR_LOOP_FINDINGS.md                 **the channel experiment: real 0/8, sham 1/8 -> E1-E3 fail**
runs/env_007/REPAIR_LOOP_PREREGISTRATION.md          arms, sham construction, bars, two-stage rule (fixed in advance)
runs/env_007/{PILOT_TERMINAL_RULE,ABLATION_FINDINGS}.md
runs/env_007/LADDER_FINDINGS.md                      the ladder result, the failure mechanism, §5 rung study
runs/env_007/LADDER_DECISION_PREREGISTRATION.md      decision rule + protocol, fixed before results
runs/env_007/train_queue.ps1                         commit-guarded training queue (spec JSON in, results out)
runs/env_007/ladder_train/launch_and_analyze.ps1     the 8 ladder trainings, then ladder_analysis.py
runs/env_007/ladder_train/README.md                  launch pattern + the page-file/commit trap
runs/env_007/control_obs_only/README.md
```

Tools added for the rung study (all at the repo root):

```
eval_pool.py                      fresh-seed evaluation of many runs -> JSON (eval_fresh_seeds prints only)
rung_analysis.py                  rho(0.6M, 1.2M), seed dispersion, pre-registered verdict
mechanistic_readout.py            the frozen component-activity readout + precision/recall/AUPRC
component_share_report.py         per-component active_rate / magnitude_share tables from summaries
advantage_scale_probe.py          per-step advantage of acting over idling (training-free diagnostic)
make_repair_variants.py           builds the oracle repair arms for L2/cand_00 (anchored, refuses blind patches)
make_align_fix_variants.py        builds the L0/cand_02 align-fix arms
make_recipe_replication.py        applies the two-edit recipe to L2/cand_14
run_repair_loop.py                drives the evidence-channel experiment (real vs sham reflection)
repair_loop_analysis.py           hit rates + Fisher + E1-E4 verdict for that experiment
make_prompt_v7.py                 derives prompts/…_v7.md from v5 by auditable replacements (+ diff)
make_overshoot_variants.py        overshoot-cliff / gentleness ablation arms
make_approach_variants.py         proximity / delta-x200 / funnel ablation arms
make_ridge_variants.py            coefficient sweep arms (x75, x100)
check_settled_stream.py           does a reward pay every step while settled? (v5 forbade this)
diagnose_ridge.py                 artifact checks: file diff, fallbacks, config, eval agreement, curves
v7_analysis.py                    V7-1/V7-4 verdicts + Fisher vs the v5-family baseline
```

---

## 7. Warnings for whoever resumes

* `prompts/eureka_01_initial_reward_v3.md` contains a **contradiction** that was later
  fixed in v4/v5 (it says "allow module-level variables" while also saying "output only
  the function body"); five v3 candidates died with `NameError`. Do not reuse v3.
* `validate_code` in `pipeline/run_03_direct_reward_generator.py` now **executes** the
  generated code (12 calls plus an episode boundary). Static-only validation silently
  passed five rewards that raised on the first step.
* `extract_code`'s no-fence fallback now preserves module-level assignments above the
  function; it used to slice them off.
* Never add `runs/env_007/passthrough_probe/reward.py` or any `ablation_probe` file to a
  population/lineage — they are diagnostics that read `info`.
* `runs/env_007/ladder_train/*` and `runs/env_007/v4_train/*` are 1.2M-step runs, not 3M.
