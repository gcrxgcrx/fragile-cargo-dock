# Pre-registration for the prompt-ladder decision (written BEFORE the outcomes exist)

Written 2026-09-20 ~05:15, i.e. **before** any of the eight 1.2M-step trainings in
`runs/env_007/ladder_train/` had produced a `training_summary.json`, and therefore before
any fresh-seed success rate for those candidates was observable. The structural check
values *were* already known at this point (they need no training); they are the
calibration set, not the test set.

Purpose: fix the decision procedure and the follow-up experiment in advance, so the
branch taken is not chosen after seeing which one looks better. The rule itself comes from
`NEXT_SESSION.md`; only the numerical bar and the follow-up protocol are fixed here.

---

## 1. The decision rule (from `NEXT_SESSION.md`, unchanged)

| `ladder_analysis.py` shows | Conclusion | Next step |
|---|---|---|
| some check has a measurable Spearman rho with fresh success | a cheap selector might exist | pre-register a threshold, then report precision / recall / AUPRC on a **fresh** candidate pool |
| every check is zero-correlation | cheap probes are closed for selection | select by ~0.6M-step short training with more seeds; keep the 60 fresh seeds permanently held out |

Checks reported by `ladder_analysis.py`: `event`, `second`, `gentle_gap`, `term_ratio`,
`hover_ratio`, `traj_succfail`. Outcome: fresh-seed success rate, seeds 30000–30059,
60 episodes, n = 8 candidates.

### The bar that makes the branch deterministic

The rule above says "measurable" without a number, which would leave room to pick the
branch after the fact. Fixed now:

* a check **qualifies** iff its rho is defined (the check has >= 2 distinct values — a
  zero-variance check makes rho undefined, DeepSeek's correction 10) **and**
  `|rho(success)| >= 0.60`;
* if several qualify, take the one with the largest `|rho(success)|`; ties broken by the
  order `event, second, gentle_gap, term_ratio, hover_ratio, traj_succfail`;
* **any** qualifying check -> **Branch A**; none -> **Branch B**.

`|rho| = 0.60` at n = 8 is a weak association (two-sided p ~ 0.12) and is deliberately not
a significance claim: it is only a trigger for spending a fresh pool on the question. No
conclusion about the method may be drawn from the ladder rho itself.

---

## 2. Branch A — the pre-registered selector, evaluated on a fresh pool

Fixed protocol, to be executed exactly as written if the bar is met.

1. **Fresh pool.** 16 *new* generations, `pilot_generate_only.py`, identical to the L2 arm
   of the ladder (`prompts/eureka_01_initial_reward_v5.md`,
   `--config configs/env007_terminal_rule_pilot.yaml`,
   `--context runs/env_007/terminal_rule_pilot/seed_0/context`, temperature 0.7,
   `deepseek-flash`), written to a **new** directory `runs/env_007/selector_pool/`.
   These are new samples, not the ladder's 16.
2. **Check.** Compute the qualifying check on all 16 with
   `analyze_terminal_dominance.py --clip 20` (seconds each). For `traj_succfail`, use
   `trajectory_ranking_check.py --clip 20` with the same 8 library seeds.
3. **Threshold.** From the ladder calibration set (n = 8) only: the threshold maximising
   Youden's J on `(check >= t) vs (success > 0)`; ties resolved by taking the **higher**
   threshold. Frozen before any pool label is computed.
4. **Training.** All 16 pool candidates, the established hyperparameters
   (`configs/env007_terminal_rule_pilot.yaml`, `n_envs=6`, clip 20, 1.2M steps, seed 0),
   via the commit-guarded queue.
5. **Labels.** Fresh-seed success on a **new** held-out block, seeds **32000–32059**
   (60 episodes). Seeds 30000–30059 are NOT used here, so they remain pristine for
   headline results — using them to label a selector would spend them.
6. **Metrics.** Precision, recall and AUPRC of `check >= threshold` as a predictor of
   `success > 0` on the pool, plus the two trivial baselines: predict-all (precision =
   prevalence, recall = 1) and random ranking (AUPRC = prevalence). Secondary:
   Spearman rho on the pool (n = 16, wide CI).
7. **Falsification.** If AUPRC <= prevalence, the check is not a usable selector: the
   cheap-selector branch closes and the conclusion falls back to Branch B's prescription.

## 3. Branch B — is the 0.6M rung a usable *selector*?

`SESSION_STATE.md` §3h measured the ~0.6M fidelity floor for the **native** reward. Whether
0.6M ranks *different candidate rewards* is untested, and that is exactly what selection
would rely on. Fixed protocol:

1. Train the **same 8** ladder candidates at 0.6M steps, seeds **{0, 1, 2}** (the 1.2M runs
   used seed 0 only), everything else unchanged. 24 runs.
2. Evaluate each on seeds **32000–32059** (same reasoning as Branch A step 5).
3. Report, over the 8 candidates: (a) seed-to-seed dispersion of success at 0.6M;
   (b) Spearman between mean-0.6M success and the already-measured 1.2M success;
   (c) the same for dock_rate and mean return; (d) whether 0.6M separates the four
   L2 (event-writing) from the four L0 candidates.
4. **Decision (fixed):** the 0.6M rung is adopted as the selection rung iff
   `rho(0.6M, 1.2M success) >= 0.60` **and** the seed-to-seed spread is smaller than the
   between-candidate spread. Otherwise the rung must be raised (next candidate: 1.0M) and
   that must be reported rather than silently using 0.6M.

---

## 3b. OUTCOME (recorded 2026-09-20 ~05:30) and the Branch B amendment

`ladder_analysis.py` finished; joined table (fresh seeds 30000–30059, 60 episodes):

| cand | event | 2nd | termRatio | gentleGap | hoverRatio | trajSucc>Fail | fresh | meanRet |
|---|---|---|---|---|---|---|---|---|
| L2/cand_00 | 1 | 1 | 1e6 | 1.140 | 1e6 | 0.773 | **0/60** | −1.41 |
| L2/cand_05 | 1 | 1 | 1e6 | 5.700 | 1e6 | 0.768 | **0/60** | −3.17 |
| L2/cand_04 | 1 | 1 | 1e6 | 1.900 | 0.25 | 0.654 | **0/60** | −0.90 |
| L2/cand_14 | 1 | 1 | 1e6 | −4.325 | 0.07 | 1.000 | **0/60** | −2.87 |
| L0/cand_02 | 1 | 1 | 1e6 | 0.000 | 1e6 | 0.268 | **0/60** | +1.55 |
| L0/cand_11 | 0 | 0 | 0 | 0.000 | 0.03 | 0.659 | **0/60** | −4.06 |
| L0/cand_13 | 0 | 0 | 0 | 0.000 | 0.05 | 0.841 | **0/60** | −1.92 |
| L0/cand_00 | 0 | 0 | 0 | 0.000 | 1e6 | 0.649 | **0/60** | −67.34 |

`outcome variance: successes = [0,0,0,0,0,0,0,0]` → **every rho is undefined**, so by the
pre-registered bar in section 1 no check qualifies: **Branch B**.

(The secondary column is interesting and is *not* the pre-registered outcome: `event`,
`second` and `term_ratio` each have rho = 0.524 against **mean return**, while success is
identically zero for all eight. Structural compliance moves the return and not the
success — a second, independent instance of the §3f lesson. No method conclusion may be
drawn from a n = 8, p ≈ 0.18 association.)

### The degeneracy, and the amendment it forces

Branch B as pre-registered in section 3 says to test the 0.6M rung by correlating its
outcome with the 1.2M outcome **over the eight ladder candidates**. That test is
**uninformative here**, and not because of a rule change: the 1.2M label vector is
`[0,0,0,0,0,0,0,0]`. A rank selector cannot be validated against a constant label — the
correlation is undefined for exactly the same reason the check correlations were.

**Amendment, fixed before any 0.6M run is launched:** run the *same* rung test on the pool
that actually carries outcome variance — the six hand-written arms, whose 1.2M fresh-seed
outcomes span 0 % → 98.3 % (`SESSION_STATE.md` §3b): `control_obs_only/train_1m2`,
`ablation_probe/probe{A,B,C,D,E}`.

Unchanged: the rung (0.6M steps), the seeds ({0,1,2}), the evaluation block
(32000–32059), the statistics (seed dispersion; rho(0.6M, 1.2M) for success and
dock_rate), and the decision bar (adopt 0.6M iff `rho >= 0.60` **and** seed-to-seed spread
< between-candidate spread).

Changed: the pool for the *rank-correlation* statistic, because the pre-registered pool's
labels are constant — a rank-degenerate label vector cannot validate a rank selector.

What was actually run (recorded after the fact, 06:20): the six arms at 0.6M with seeds
{0,1,2} (18 runs, the ranking test in §3b), **and** the eight ladder candidates at 0.6M with
seed 0 (8 runs). The latter are not part of the ranking statistic — their 1.2M labels are all
zero, so they can only serve as a *negative class*, which is exactly how they are used by the
readout test in §3c. Running them costs one wave and makes that negative class measured at
0.6M rather than assumed.

Secondary benefit: re-measuring those six arms at 1.2M on a **new** seed block
(32000–32059, unused) is an independent replication of §3b on different seeds.

## 3c. Pre-registered mechanistic readout at the 0.6M rung (written 05:37, before any 0.6M summary was read)

Induced from the **1.2M** component tables (`component_share_report.py`) and fixed here
before the 0.6M runs are read. At the time of writing the first wave of 0.6M runs may
already have written `training_summary.json` to disk; none of them had been opened.

**The observation that induces it (1.2M, all labels known):** every arm with a positive
fresh-60 outcome carries ~96 % of its reward mass in a *sparse-active, positive* term
(`probeA` `dock_settled_hold` share 0.966 / active_rate 0.027 / +180 per episode;
`probeB` `terminal_success` 0.861 / 0.00055 / +60; `probeC` 0.964 / 0.00484 / +300;
`probeD` 0.958 / 0.029 / +140; `probeE` `success_event` 0.960 / 0.00304 / +255). Every arm
with a zero outcome lacks that structure: `control v1` and `L0/cand_{02,11,13}` have only
**dense** positive terms (active_rate 1.0, farmable — `L0/cand_02` collects +311 per
episode and still never satisfies the success predicate), `L0/cand_00` and `L2/cand_00`
carry their mass in a *negative* sparse term, `L2/cand_04` in a dense penalty,
`L2/cand_14` in a positive sparse term that is far too small (+0.82 per episode), and
`L2/cand_05` in nothing at all.

**Readout, frozen.** For a trained run, from
`external_eval.final_policy_component_evaluation`:

* `sparse` := components with `0 < active_rate < 0.05`;
* among `sparse`, take the one maximising `magnitude_share` **subject to
  `episode_sum_mean > 0`**; call it `c*` (if none exists, predict NEGATIVE);
* predict **POSITIVE** iff `magnitude_share(c*) >= 0.50` **and**
  `episode_sum_mean(c*) >= +20.0`.

Per candidate, aggregate over its runs by **majority vote** (3 seeds for the six
hand-written arms, 1 seed for the eight ladder candidates).

**Labels** (1.2M, seeds 30000–30059, replicated for the six arms on 32000–32059):
POSITIVE = `probeA`, `probeB`, `probeC`, `probeD`, `probeE` (all > 0 %);
NEGATIVE = `control_v1` plus all eight ladder candidates (all exactly 0/60).
Prevalence = 5/14 = 0.357.

**In-sample result at 1.2M (NOT evidence):** 14/14 correct. Three thresholds on fourteen
points is an in-sample fit; it is reported only to show the readout is well defined.

**The test, and the bar.** Evaluate the readout on the **0.6M** runs of the same 14
candidates (out-of-sample in budget, and in seed for the ladder candidates) against those
1.2M labels. Report precision, recall and AUPRC, with the predict-all-positive baseline
(`precision = 0.357, recall = 1.0, AUPRC = 0.357`).

The mechanistic readout is adopted as a selector **iff `AUPRC >= 0.60` and
`recall >= 0.60`** — both fixed now, before the 0.6M numbers are read. Otherwise the
conclusion is that the cheapest honest instrument must be longer than 0.6M, and the
outcome-based rung result in section 3 governs.

**Ranking companion for AUPRC** (fixed in the same edit, still before any 0.6M summary was
read): `score = magnitude_share(c*) * episode_sum_mean(c*)`, and `0.0` when no positive
sparse term exists. At 1.2M this scores the five positives at 51.7–289 and every negative
at <= 0.82, so the binary thresholds above are a cut on this score.

## 3d. Pre-registered intermediate rung: does the readout recover by 1.0M? (written 06:18)

Result so far: the frozen readout (§3c) separates **14/14 at 1.2M** (in-sample) and, out of
sample, gives **precision 1.000 / recall 0.400 / AUPRC 0.877** at 0.6M → rejected on recall.
The per-seed detail says why: at 0.6M the terminal term is still in transit (`probeC` seed 0
dominant, seeds 1–2 stuck at `dock_enter`; `probeD` 1/3 seeds; `probeE` 1/3 seeds).

**Pre-registered prediction, before any 1.0M run exists:** if that diagnosis is right, the
readout recovers at an intermediate rung. The six hand-written arms are retrained at
**1.0M** with seeds {0,1,2} (18 runs, same config, same fresh block for any outcome
measurement), and the readout is applied with the same frozen thresholds and the same
majority-over-seeds aggregation.

* **Bar (fixed now):** the readout is usable at 1.0M iff it certifies **at least 3 of the 5
  positive arms** (recall ≥ 0.60) **and** does not certify `control_v1` (precision 1.0 on
  this pool). Both conditions, as at 0.6M.
* Falsified if recall < 0.60 or if `control_v1` is certified. If it is falsified, the rung
  that works is 1.2M — the same budget as simply training the candidate — and the
  cheap-selector programme is closed at every rung tested (0.6M, 1.0M, 1.2M).
* Scope note: only the six arms are run at 1.0M. The eight LLM candidates already show **no
  active sparse term at 0.6M and at 1.2M** and 0/60 at both, so they are carried as context
  rather than re-run; the test statistic is recall on the five positives plus the
  `control_v1` false-positive check.

**OUTCOME (recorded 07:05).** Readout at 1.0M: precision 0.833, **recall 1.000**, AUPRC
0.877 (baseline 0.833) — all five positives certified, which confirms the mid-transit
diagnosis — **but `control_v1` is certified positive in 3/3 runs**, so the pre-registered
condition is violated and the verdict is **REJECT** (`readout_10m_all.json`). The
outcome-based rung was measured as well, to keep both instruments comparable:
`rho(1.0M, 1.2M success) = +0.371` (bar ≥ +0.60) with within-candidate seed sd 0.252 ≈
between-candidate sd 0.289 → **REJECT** (`rung_analysis.json` with `--short-label 1.0M`).

Why it fails at 1.0M rather than 0.6M is a different reason, and worth stating: `control_v1`
contains a `+20`/step settled *stream*, and at 1.0M that stream is active and dominant
(share 0.971, +238 per episode) in every seed, so the readout cannot separate a farmable
stream from the probes' one-off settled *event*. At 1.2M the same arm is rejected only
because its policy has abandoned docking by then, which switches the term off — i.e. the
readout's 14/14 at 1.2M depends partly on that degeneration. Full discussion:
`LADDER_FINDINGS.md` §5.6–5.7.

## 4. Standing constraints (unchanged, but restated because they are easy to violate)

* Seeds 30000–30059 stay held out for headline numbers; the selector studies above use the
  new block 32000–32059.
* Never put `runs/env_007/passthrough_probe/reward.py` or any
  `runs/env_007/ablation_probe/*` reward into a pool, lineage or elite set — they read
  `info` and are diagnostics.
* Only `deepseek-flash` for generation, `DEEPSEEK_THINKING=disabled`, key never written to
  a file and never handed to a subagent.
* Long runs are harness background jobs or detached `Start-Process`.

## 5. Resource note

Launching 8 trainings at once is **not** safe on this machine: the commit limit (page file)
is the binding constraint, and workers die during `import torch` with
`WinError 1455 / 页面文件太小` while the parent hangs at 0 CPU. See
`runs/env_007/ladder_train/README.md`. All training here goes through the queue in
`runs/env_007/ladder_train/launch_and_analyze.ps1` (`-MaxParallel 4` or fewer).
