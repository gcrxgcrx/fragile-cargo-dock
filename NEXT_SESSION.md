# NEXT SESSION — start here

This file is the entry point after a context reset. Read it first, then do what
"First actions" says. Everything else is indexed at the bottom.

---

## What this project is

We are running **reward-search experiments** on a custom RL environment,
`FragileCargoDock-v0` (top-down Box2D, cart pushes a fragile crate into a dock; the cart
has no brake, so it must release early and let floor drag carry the crate in).

The method under test is **CREATE** (called **DERES** in the paper's tables — same
pipeline, same runs); the baseline is an **EUREKA-style population search**
(`pipeline/run_eureka_population.py`).

**Scope: experiments only.** The paper is not ours. All paper-reporting issues are
out of scope and are listed in `HANDOFF_QUESTIONS.md` section 6 so they are not
re-litigated.

Repo: `D:\Code\python\research\form_github\expert-reward-agent`
Interpreter: `D:\Code\python\research\llm_env_310\Scripts\python.exe`
(bare `python` is msys64 Python 3.14 with **no packages** — never use it).

---

## First actions

1. **Read `SESSION_STATE.md`.** It is the full resume document: environment physics,
   harness facts, every established measurement, the ten errors I already falsified
   (so they are not re-derived as conclusions), and the warnings.
2. **Where the project stands, in six lines** (all measured; details in `SESSION_STATE.md` §5):
   * the ladder decision fired **Branch B** — cheap structural probes are closed for selection
     (eight candidates spanning every check, all 0/60);
   * **the two-edit repair works**: on two independent LLM candidates, (i) the candidate's own
     delta-progress coefficient raised, (ii) its one-off terminal payoff converted into a dense
     per-step payoff on the **instantaneous success predicate** → 0/60 becomes 23/60 and, at a
     better coefficient, **43/60 = 71.7 %** — the project's best result ever
     (`REPAIR_TEST_FINDINGS.md`);
   * **the evidence channel is not the bottleneck**: the pipeline's own reflection (which does
     contain the decisive activation evidence) gave real **0/8** vs sham **1/8** repairs
     (`REPAIR_LOOP_FINDINGS.md`);
   * **v7** (removing the scaffold rule that *forbade* the per-step payoff) made 8/8 write the
     stream and produced the first unedited LLM candidate ever to enter the dock, but only
     **1/8 hits, 3/60 = 5 %** (`V7_PROMPT_PREREGISTRATION.md`);
   * **the difficulty is SETTLING, and it is near-coin-flip** — §5.8: `dock_entered` is set fairly
     reliably by the coefficient while "does it complete" varies wildly across seeds; the same
     reward can dock in 95 % of episodes and score 0/60;
   * **nine single-axis "guidance" edits fall to 0/60** against a matched baseline, including every
     direction a domain expert reaches for first (`OVERSHOOT_ABLATION_…`, `APPROACH_ABLATION_…`).

   Nothing above needs re-running; the raw JSONs are on disk.

3. **The single most important thing to internalise: the reward is a two-factor problem.**
   **Reachability** (does the policy get the crate to the dock) is close to a function of the
   reward's scale; **settling** (does it complete the 10-step hold) is near-coin-flip at fixed
   settings — measured ×100 (coefficient 1200): `dock_entered` **0.95 / 0.78 / 0.95** but success
   **31 / 43 / 0** out of 60. Consequences, both load-bearing:
   * **single-seed candidate comparisons are one lottery draw.** Report `dock_entered` and success
     **separately**, and state the seed count. Any "0/60 vs 1/60" contrast is inside the noise.
     Before quoting the paper's or the repo's per-candidate numbers, **check how many seeds they
     used** (`multi_seed.num_seeds` is 3 in the CREATE base config; EUREKA side unverified).
   * the earlier phrasing "the working reward is a narrow ridge" is **superseded by §5.8**: it is a
     non-monotone *probability of landing in the good attractor*, plus settling variance. The
     artifact checks that rule out bugs/config drift/eval artefacts (`diagnose_ridge.py`) are in
     §5.8 — read them before re-litigating whether the jaggedness is real.

4. **v7 prompt experiment (DONE, scored 01:45) — one line for context.** v7
   (`prompts/eureka_01_initial_reward_v7.md`) removed the scaffold rule that *forbade* the
   ingredient the repair needs (v5 line 112 banned per-step reward on the success state). The
   stream form then appeared in **8/8** candidates and **`cand_01` became the first unedited LLM
   candidate ever to enter the dock** (`dock_entered` 0.18, where all 35+ earlier ones were 0.00)
   — but the honest rate is **1/8 hits, 3/60 = 5.0 %**, Fisher p = 0.333 vs the 0/16 baseline:
   inside the noise floor. Details: `SESSION_STATE.md` §5.5, `V7_PROMPT_PREREGISTRATION.md`.

5. **THE NEXT STEP — user's directive of 03:00.** Write the **v8** prompt (`prompts/…_v8.md`,
   derived from v7 with `make_prompt_v7.py`-style auditable replacements) that prescribes the
   **form** and nothing else:

   * **require** a dense per-step payoff on the **instantaneous success predicate** (inside AND
     aligned AND slow) — this is the ingredient v5 forbade and v7 restored;
   * **require** a delta-progress term on the candidate's own distance-reduction signal (a
     *delta*, not a proximity/state reward) and state the two-condition requirement: per-step
     reward of acting must beat idling, and the settled state must be the argmax;
   * **leave the numeric coefficient to the generator** — do **not** hard-code ×50/×100. The
     measured surface is non-monotone in that scalar (2 % at ×75, 41 % at ×100, 0 % at ×200) and
     the user's decision is that the LLM should explore it;
   * **must not** include any of the refuted ideas: no positional overshoot cliff, no extra
     gentleness stacked on an existing closing-speed penalty, no proximity/state reward, no
     "near AND slow" funnel;
   * evaluate candidates with **more than one seed** and report `dock_entered` and success
     separately.

   **Do not build an `A > 0` mechanical gate and call it a fix.** `A` is smoothly monotone in the
   coefficient while the outcome is not: it scores ×200 (0/60) highest and ×75 second-lowest. It can
   certify a *necessary* condition only (`ADVANTAGE_PROBE_FINDINGS.md`, `SESSION_STATE.md` §5.7).

   **Then, prereg'd and not yet driven — the ridge-recovery operator test**
   (`runs/env_007/RIDGE_RECOVERY_PREREGISTRATION.md`): seed each method's operator with an
   off-ridge candidate whose failure cause is **known** and see whether it recovers delivery.
   EUREKA side = `build_reward_reflection` + `materialise_reward(mode="edit")` (as in
   `run_repair_loop.py`); CREATE side = `pipeline/run_04_build_iteration_context.py` (context +
   diagnosis JSON) then `pipeline/run_05_reward_revision.py --previous-reward … --iteration-context …`.
   Key handling: read line 19 of
   `runs/env_001/ablation_eureka_feedback_v4/reproduction/run_ablation_score_only_v4.ps1` at
   runtime, export it as `EUREKA_DEEPSEEK_API_KEY`, and set `DEEPSEEK_THINKING=disabled`; never
   print it or write it to a file. Costs are in `runs/env_007/COMPUTE_LEDGER.md`.

6. **Then pick up at the open questions — both earlier hypotheses are settled or closed:**

   * **H1 — generator strength: CLOSED by the user.** A stronger generator's API is not
     available (stated 2026-09-20 evening). Do not plan around it.
   * **H2a — "the evidence channel is the problem": MEASURED NEGATIVE.** The pipeline's own
     reflection channel — which verifiably contains `success_event` activation = 0 % for the
     L2 candidates and 100 %-active dense terms for the L0 ones — was given to the model with
     a permuted-table sham control, 8 repairs per arm. **real 0/8 vs sham 1/8**, Fisher
     p = 1.0000, E1–E3 all fail. Pre-registered consequence: **adding channels, including
     P5's trajectory-evidence extension, is predicted to be worthless on this environment**
     — the channel already carries the information and the operator still does not produce
     the repair. Write-up: `runs/env_007/REPAIR_LOOP_FINDINGS.md`. Do not rebuild H2a
     without first explaining why one round with this operator was not a fair test.
   * **H2b — what is still open.** (i) **Iteration**: repair → retrain → reflect again, more
     than one round (untested; the loop here ran one round). (ii) **Operator**: the same
     evidence was sufficient *for a human* — the oracle two-edit recipe hits 2/2 targets at
     23/60 while the model's own edits hit 0/8 — so the gap is in the operator, and a
     different prompt shape (e.g. asking explicitly for the two named conditions below) is
     the cheapest remaining probe. (iii) **The L0 family's repair** is untested.

   **The target any such loop must reach is measured** (the session's main positive result,
   `runs/env_007/REPAIR_TEST_FINDINGS.md`): **two localized edits** — strengthen the candidate's
   own *delta*-progress term, and convert its one-off terminal payoff into a dense per-step payoff
   on the **instantaneous success predicate**. That took two independent LLM candidates from 0/60
   to 23/60 each, and with the coefficient set to 1200 (×100) one seed reached **43/60 = 71.7 %**,
   the best number this project has produced. The *form* is what is known; **the coefficient is
   deliberately left to the LLM** (§5.8, user decision).

   The selection fallback that used to be listed here — short training at the **~0.6M**
   fidelity floor with more seeds — **has now been measured and rejected**; do not spend
   budget re-testing it. On six hand-written arms with known 1.2M outcomes (0 % → 98.3 %),
   the 0.6M rung gives `rho = −0.486` against 1.2M success (it ranks the 0 %-at-1.2M control
   *second of six*) and its seed noise (sd 0.236) exceeds the between-candidate spread
   (0.149). A component-activity readout that separates 14/14 at 1.2M also fails at 0.6M
   (precision 1.00, **recall 0.40**). More seeds cannot fix a biased rung. The intermediate
   **1.0M** rung was then tested too, with the bar fixed in advance: outcome `rho = +0.371`
   (still short of 0.60, and `control_v1` still ranked 4th of 6), and the readout reaches
   recall 1.00 but **certifies `control_v1` in 3/3 seeds** → rejected as pre-registered.
   Details and raw data: `runs/env_007/LADDER_FINDINGS.md` §5 (verdict table in §5.7),
   `runs/env_007/{rung_06m,rung_10m}/`.
   **Selection cannot be made cheaper than evaluation on this environment** — the two
   hypotheses below are generation-side and are where the remaining work is.

---

## Hard constraints (violating these invalidates the work)

* **API keys.** `EUREKA_DEEPSEEK_API_KEY` for the generation/EUREKA flow,
  `DEEPSEEK_API_KEY` for CREATE. The key must **never** be handed to a subagent or to any
  task outside those two flows, and must **never** be written to a file. It can be read at
  runtime from
  `runs/env_001/ablation_eureka_feedback_v4/reproduction/run_ablation_score_only_v4.ps1`
  (line 19) instead of being retyped into the transcript.
* **Only `deepseek-flash`** (DeepSeek-V4.1-Flash) for generation. `deepseek-v4.1f`
  returns HTTP 400. `DEEPSEEK_THINKING=disabled` must be exported; thinking mode ignores
  `temperature` and can return an empty `content` with `finish_reason='length'`.
* Long runs must be harness `run_in_background: true` jobs or detached `Start-Process`,
  otherwise they are killed when the spawning shell scope is torn down.
* **Do not launch 8 trainings at once.** The binding resource is the Windows **page file /
  commit limit**, not RAM. Free physical memory can look healthy (12.9 GB) while free commit
  is 6.7 GB; the `SubprocVecEnv` workers then die during `import torch` with
  `OSError: [WinError 1455] 页面文件太小` / `WinError 1114 ... shm.dll`, and the parent
  trainer **hangs at 0 CPU** instead of erroring — it looks like a stall, and the only
  proof is in the `.err.log`. Use `runs/env_007/train_queue.ps1` (or
  `ladder_train/launch_and_analyze.ps1`) with `-MaxParallel 4` or lower. Background:
  `runs/env_007/ladder_train/README.md`.
* Never add `runs/env_007/passthrough_probe/reward.py` or any `runs/env_007/ablation_probe/*`
  file to a population, lineage or elite set — they are diagnostics that read `info`.
* `runs/env_007/{ladder_train,v4_train}/*` are **1.2M**-step runs, not 3M.
* The 60 fresh seeds **30000–30059** are the honest evaluation set and must stay held out
  from any selection.

---

## What is already settled (do not redo)

* The harness is fine: the native reward scores **96.8 %** bypassing the wrapper and
  **90 %** *through* it at the default per-step clip of 20. Raising the clip to 600 made
  the best LLM candidate worse.
* The observation-only contract **can** express a solvable reward: hand-written arms reach
  40 % / 45 % / **73.3 %** / **65 %** / **98.3 %** on fresh seeds, versus 0 % for the
  un-augmented control.
* No LLM-produced reward has ever worked: best ever **5/60 = 8.3 %**, and that candidate
  collapses to 0/60 at full budget. Two-sided Fisher p = **0.0573**, so even that was not
  significant. Rule of three puts the 95 % upper bound of `0/60` at **5 %**.
* **The prompt ladder is the key new result** (generation only, 16 candidates per level):
  removing my scaffold rules drops "writes a one-off terminal event" from 16/16 to 0/16
  (L1, the paper's own prompt) and 1/16 (L0, interface-only), and gentleness from 15/16 to
  0–2/16. L0 ≈ L1, so the leak is specifically the rules **I** added, not the paper's prompt.
* **Ranking correctness ≠ learnability.** `trajectory_ranking_check.py` shows the 0 %
  control reward orders success above every failure on physically reachable trajectories
  with a **perfect** 1.000 accuracy, exactly like the 73.3 % probe. So no training-free
  probe can be a positive selector — this is measured, not argued.
* **The structural checks carry no information about success.** Eight further candidates,
  chosen to span the checks from "all four pass" to "none pass", were trained at 1.2M and
  **all eight scored 0/60** (table in `SESSION_STATE.md` §5). With the earlier ones this is
  **11 scaffold-compliant LLM rewards with zero deliveries**. Meanwhile hand-written
  rewards on the same contract reach 40 % / 45 % / 65 % / 73 % / 98 %.
* **Why they fail — the mechanism, measured.** Every arm that works puts ~96 % of its reward
  mass in a **positive, sparse (one-off) term that is actually reached**; every LLM arm puts
  **0 %** there. Its mass sits instead in a penalty, in a **farmable dense term** (`L0/cand_02`
  collects +311 per episode and still never docks), or nowhere at all (`L2/cand_05`'s reward
  is identically 0 on the trajectory its policy produces → no gradient). Two named failure
  modes; tables in `runs/env_007/LADDER_FINDINGS.md` §2. The structural checks probe the
  *function* on states a scripted expert reaches and never ask whether the term is *active
  under the policy's own trajectory* — which is why they cannot predict learning.
* **Longer training can destroy a learned behaviour.** `control_v1` reaches **39/60 at 0.6M
  (seed 0) and 0/60 at 1.2M**, while all five hand-designed probes *improve* (C 25→59, D 0→44,
  E 0→46). Reward hacking emerges with optimisation; this is also what §3c's 5/60→0/60 was.
* **The LLM failure is two-layer, and a two-edit repair reaches 38.3 %.** Two independent
  LLM candidates that both score 0/60 each reach **23/60 on fresh seeds** after (1) their own
  approach coefficient ×50 and (2) their one-off terminal payoff converted into a **dense
  +20/step stream** on the instantaneous success predicate. Neither edit alone works: the
  scale edit docks but cannot settle, the stream edit is *bit-for-bit* equivalent to the
  untrained copy because the stream's gate is never reached. Adding gentleness **hurts**
  (0/60) and raising the candidate's own speed penalty ×100 makes it worse (0/60).
  `success_event` activation goes 0.0000 → 0.0012/0.0017. This is an **oracle** repair:
  it measures the size of the target, not search discoverability.
  Write-up: `runs/env_007/REPAIR_TEST_FINDINGS.md`.
* **Why they fail at all — a sign problem.** On scripted trajectories the base reward already
  orders correctly (success +56, pushing +6.4, idle −0.8 per episode) yet its policy sits
  still, because the **per-step advantage of acting over idling is only ≈0.018**; the
  advantage-scale probe measures `A = −0.259/step`, i.e. *acting is worse than nothing*, so
  the policy is optimising its reward correctly. Learnability needs **both `A > 0` and the
  reward's argmax being the success behaviour**; `advantage_scale_probe.py` measures only the
  first (AUPRC 0.413 as a selector → rejected, it ranks dense farmable rewards highest).
* **An inverted alignment term, found and fixed, still does not train.** `L0/cand_02` computes
  `1 − |obs[10]|` while `DOCK_ANGLE = 0`, so it rewards holding the crate **perpendicular**
  (a state that can never succeed) — reward hacking with a one-token root cause. The
  one-subscript fix moves the training-free ordering check **0.038 → 0.702** and the arm still
  trains to 0/60, because its advantage stays at `A ≈ −0.0004`. A second, stronger instance of
  §3f, and a warning: repairing the ordering is not repairing the reward.
* **The reward is a two-factor problem, and settling is the binding one.** A candidate's
  `dock_entered` is set fairly reliably by the reward's scale; whether it *completes the 10-step
  hold* is near-coin-flip at fixed settings (measured ×100: dock 0.95/0.78/0.95, success 31/43/0
  out of 60). So **never compare candidates on a single seed**, and always report `dock_entered`
  and success separately. `SESSION_STATE.md` §5.8.
* **Nine single-axis "guidance" edits destroy a working reward**, including every direction a
  domain expert reaches for first: a positional overshoot cliff (−20 and −200/step), an extra
  gentleness penalty, a proximity (state) reward, and a "near AND slow" funnel — all 0/60 against
  a matched baseline, with the mechanism (a positional cliff punishes the only route to the dock,
  so `A` flips negative and the optimum becomes *not approaching*). Do not re-propose them without
  new evidence. `runs/env_007/{OVERSHOOT,APPROACH}_ABLATION_PREREGISTRATION.md`.
* **The advantage probe `A` is a necessary-condition diagnostic, not a selector.** It is smoothly
  monotone in the reward's coefficient while the outcome is not (it ranks the 0/60 ×200 arm
  highest). It answered a real question — `A ≤ 0` means *inaction is optimal*, which is exactly the
  whole v5/v7 immobile family — but a gate built on it would pass arms that score 0/60.
  `runs/env_007/ADVANTAGE_PROBE_FINDINGS.md`, §5.7.
* **The existing evidence channel does not enable repair (measured, with a sham control).**
  Real reflection (which shows `success_event` activation = 0 % / dense 100 %-active terms) vs
  the same report with its component tables shuffled: **real 0/8, sham 1/8**, Fisher p = 1.0.
  The single hit is in the sham arm and equals the old best-ever LLM luck (5/60). Pre-registered
  reading: the ceiling here is the **operator, not the evidence** — so do not invest in new
  channels (including P5) before testing iteration or a different prompt shape.
* **Evaluation-noise floor.** Re-measuring the six arms on a different 60-seed block moves an
  individual number by up to **11.7 points** (`probeE` 65.0→76.7, `probeB` 45.0→36.7) while
  the ordering stays put. Compare candidates **within one seed block**; do not read a few
  points as signal.
* **Consequence — corrected by this session's repair result.** Across ~35 one-shot LLM rewards
  none delivers the crate, and neither does any cheap selector. **But the ceiling is not the
  contract or the harness**: two of those candidates reach 23/60 and 43/60 after two localized
  edits, and the *ingredient the scaffold forbade* (a per-step payoff on the success state) is
  exactly what they needed. So the open question is no longer "can a reward be written that
  works" — it can — but "**can the generator or the loop produce that form unaided**", which is
  what v8 + the ridge-recovery test are for. The generator never
  emits a reward whose global landscape PPO can follow.
* **EUREKA's reflection already contains per-component activation rate**
  (`run_eureka_population.py:245-248`), despite its own docstring claiming otherwise.
  Therefore "evidence asymmetry" is a **joint** property of environment + channel, not an
  environment property. `docs/external_reviews/` holds both external answers; their
  disagreements are adjudicated in `SESSION_STATE.md` section 5.

---

## Document index

| file | what it is |
|---|---|
| `SESSION_STATE.md` | **the resume document** — read this second |
| `SESSION_STATE.md` §3e | the prompt-ladder dose-response table |
| `SESSION_STATE.md` §3f | the trajectory-ranking result (ranking ≠ learnability) |
| `SESSION_STATE.md` §3g | the EUREKA evidence-channel fact |
| `SESSION_STATE.md` §4 | the ten errors already falsified |
| `SESSION_STATE.md` §7 | warnings: stale v3 prompt, validator/extractor fixes, what not to reuse |
| `HANDOFF_QUESTIONS.md` | the P1–P5 problem statement, with the scope split |
| `docs/external_reviews/chatgpt_answers_P1-P5.md` | external answer #1 (verbatim) |
| `docs/external_reviews/deepseek_v41f_answers_P1-P5.md` | external answer #2 (verbatim, includes 14 self-corrections) |
| `runs/env_007/ABLATION_FINDINGS.md` | the single-variable ablation write-up |
| `runs/env_007/PILOT_TERMINAL_RULE.md` | the first pilot write-up |
| `runs/env_007/control_obs_only/README.md` | the hand-written control and its result |

Tools, in the order they are useful:

| tool | purpose |
|---|---|
| `measure_release_ballistics.py` | the lead-time law: coast = `(v0 − 0.05)/2`, margin = `3.6/v0` steps |
| `analyze_terminal_dominance.py --clip 20 --summary REWARD.py...` | four structural checks per reward, seconds each |
| `trajectory_ranking_check.py --clip 20 [--describe] REWARD.py...` | reachable-trajectory ordering check |
| `diagnose_control.py RUN_DIR` | per-episode failure mechanism of a trained policy |
| `eval_fresh_seeds.py --episodes 60 --seed-offset 30000 RUN_DIR...` | honest scoring on unseen seeds |
| `pilot_generate_only.py` | generation-only A/B, N candidates per prompt, no training |
| `ladder_analysis.py` | joins checks with outcomes, per-check Spearman |
| `eval_pool.py` | fresh-seed scoring of many runs → JSON (unlike `eval_fresh_seeds.py`, which only prints) |
| `rung_analysis.py` | is a short-training rung a valid selector? rho vs full budget + seed dispersion + verdict |
| `mechanistic_readout.py` | frozen component-activity readout (sparse positive term dominating) → precision/recall/AUPRC |
| `component_share_report.py` | per-component `active_rate` / `magnitude_share` table for any set of `training_summary.json` |
| `runs/env_007/train_queue.ps1` | commit-guarded training queue: takes a spec JSON, staggers starts, kills and retries hangs |
