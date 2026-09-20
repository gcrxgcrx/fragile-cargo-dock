# REPAIR-LOOP FINDINGS — does the evidence channel that already exists suffice?

Pre-registration (arms, channel, sham construction, N, bars, predictions, two-stage rule):
`runs/env_007/REPAIR_LOOP_PREREGISTRATION.md`, fixed before any repair was generated.
Driver: `run_repair_loop.py`; operator and report are the pipeline's own
(`materialise_reward(mode="edit")` + `build_reward_reflection`), not a new channel.

## 1. Setup, verified before generating anything

Eight targets, one repair per target per arm: the four `L2` candidates (weak, gated advantage:
`A` = −0.0004 … −0.259) and the four `L0` candidates (dense terms paying for the wrong
behaviour: `A` up to +0.394, never succeeds).

The **real** channel is `build_reward_reflection(<target's 1.2M training_summary.json>)`. Its
content was inspected *before* generation, and it does contain the decisive evidence:

* `L2_cand_00`: `success_event` and `enter_dock_event` at **0.0 % activation in every training
  window**, while the active terms sit at 0.1–1.5 % — i.e. "my terminal events never fire" is
  readable straight off the table;
* `L0_cand_02`: `crate_docking_quality` at **100 % activation from the first window**, with
  `crate_to_dock_progress` rising 2.4 % → 52 % — the farmable-dense signature.

The **sham** arm gets the byte-identical report (same length, same task-score block, same
tables, same marginals) with every per-component table's **rows shuffled and each value column
shuffled independently**, destroying the name↔value correspondence and the magnitude ordering.
Verified: 3 component tables shuffled per report, all other blocks identical.

Operator identical in both arms: `prompts/eureka_02_reward_edit.md`, `deepseek-flash`,
T = 0.7, same context, `max_validation_retries=3`. The key was read at runtime from line 19 of
the existing reproduction script, exported under the name the generation flow expects, never
printed and never written to a file.

**16 repairs generated, all 16 valid** (8 real, 8 sham; one needed a validation retry after
`NameError: _PREV_T`).

### What the model actually did (read before any outcome)

The repairs are **not** reproductions of the two-edit recipe from
`REPAIR_TEST_FINDINGS.md`. `real/L2_cand_00` keeps its one-off streak bookkeeping and instead
rewrites the shaping (`crate_to_dock_progress = 4.0 * progress_signal`, plus a new dense
`docking_quality = 2.0 * joint * bounds_gate`); `real/L2_cand_05` and `real/L2_cand_14` keep
their 10-step one-off events; `real/L0_cand_02` is rewritten into a compact dict-literal form.
So any success below is the **model's own** repair, chosen by it from the evidence — which is
the point of the experiment.

### Training-free readout of the 16 repairs (secondary, pre-registered as E4's companion)

`advantage_scale_probe.py`, `A` per step:

| real arm | `A` | | sham arm | `A` |
|---|---:|---|---|---:|
| `L2_cand_05` | **+2.130** | | `L2_cand_04` | +6.626 |
| `L0_cand_13` | **+1.594** | | `L2_cand_05` | +5.194 |
| `L2_cand_14` | +0.110 | | `L0_cand_00` | +0.967 |
| `L0_cand_11` | −0.208 | | `L0_cand_02` | +0.783 |
| `L2_cand_00` | −0.219 | | `L0_cand_13` | +0.234 |
| `L0_cand_02` | −0.253 | | `L0_cand_11` | +0.112 |
| `L0_cand_00` | −0.338 | | `L2_cand_14` | +0.034 |
| `L2_cand_04` | −6.140 | | `L2_cand_00` | −0.223 |

Real arm: 3 of 8 repairs reach `A > 0`; sham arm: 6 of 8. As established in
`ADVANTAGE_PROBE_FINDINGS.md`, `A` cannot tell "paid to do the task" from "paid to do
anything", so this is context, not a discriminator.

## 2. Outcome (fresh seeds 33000–33059, 60 episodes — a block used for the first time here)

Protocol: 1.2M steps, `n_envs=6`, clip 20, seed 0, via the commit-guarded queue. Raw:
`runs/env_007/repair_loop/eval_all_block33000.json`, `analysis.json`.

| arm | target | fresh-60 | dock | mean return | `A` (training-free) |
|---|---|---:|---:|---:|---:|
| real | `L2_cand_00` | 0/60 | 0.00 | −2.32 | −0.219 |
| real | `L2_cand_05` | 0/60 | 0.00 | −3.45 | +2.130 |
| real | `L2_cand_04` | 0/60 | 0.00 | −3.87 | −6.140 |
| real | `L2_cand_14` | 0/60 | 0.00 | −1.19 | +0.110 |
| real | `L0_cand_02` | 0/60 | 0.00 | −3.88 | −0.253 |
| real | `L0_cand_11` | 0/60 | 0.00 | −6.43 | −0.208 |
| real | `L0_cand_13` | 0/60 | 0.00 | +1.10 | +1.594 |
| real | `L0_cand_00` | 0/60 | 0.00 | −3.96 | −0.338 |
| sham | `L2_cand_00` | 0/60 | 0.00 | −1.10 | −0.223 |
| sham | `L2_cand_05` | 0/60 | 0.00 | −10.15 | +5.194 |
| sham | `L2_cand_04` | 0/60 | 0.02 | +2.43 | +6.626 |
| **sham** | **`L2_cand_14`** | **5/60 = 8.3 %** | **0.55** | **+30.71** | +0.034 |
| sham | `L0_cand_02` | 0/60 | 0.00 | −3.45 | +0.783 |
| sham | `L0_cand_11` | 0/60 | 0.00 | −2.51 | +0.112 |
| sham | `L0_cand_13` | 0/60 | 0.00 | −1.14 | +0.234 |
| sham | `L0_cand_00` | 0/60 | 0.00 | −99.77 | +0.967 |

**hits: real 0/8, sham 1/8**; one-sided Fisher (real > sham) = **1.0000**.

## 3. Verdict against the pre-registered predictions

| # | prediction | outcome |
|---|---|---|
| **E1** | real arm produces ≥ 1 success | **FAILS** — 0/8, and **none of the eight repairs ever entered the dock** (`dock_entered` 0.00 for all of them) |
| **E2** | real > sham | **FAILS** — p = 1.0000; the only success in the whole experiment is in the **sham** arm |
| **E3** | sham hit rate ≈ 0 | **FAILS** — sham produced the single hit (`sham/L2_cand_14` = 5/60 = 8.3 %) |
| **E4** | successes accompanied by `A > 0` | **NOT SUPPORTED** — `A > 0` in 3/8 real repairs (0 successes) and 7/8 sham repairs (1 success); `A` neither suffices nor predicts |
| stage 2 | run only if 0.05 ≤ p < 0.20 | not triggered (p = 1.0); the experiment stops at one stage |

### Reading it honestly

1. **The load-bearing result is E1's failure.** The real channel demonstrably contains the
   decisive evidence — inspected *before* generation: `success_event` and
   `enter_dock_event` at **0.0 % activation in every window** for `L2_cand_00`, and
   `crate_docking_quality` at **100 % activation from the first window** for `L0_cand_02`.
   Handed exactly that, the model's eight repairs scored 0/60 and did not even reach the
   dock, while the oracle two-edit repair reaches `dock_entered` **0.97** and 23/60. So the
   evidence was available and was not converted into the required edit.
2. **My stated prior was wrong.** I predicted the real table would beat the sham because it
   contains both signals. It did not: real 0/8 vs sham 1/8. That prediction and its failure
   are recorded here rather than quietly dropped.
3. **The one hit is in the sham arm, and it is exactly the historical luck magnitude.**
   5/60 = 8.3 % is the same number as the project's previous best LLM candidate
   (`SESSION_STATE.md` §3c), which itself collapsed to 0/60 at full budget and was never
   significant (two-sided p = 0.0573). It is also the size of hit one expects from
   occasionally lucky one-shot generations, and the sham arm's repairs had the *largest*
   per-step advantage values of the whole experiment (`sham/L2_cand_04` = +6.6,
   `sham/L2_cand_05` = +5.2), i.e. they wrote dense, farmable terms — the very pattern that
   `LADDER_FINDINGS.md` §2.1 identifies as a failure mode.
4. **Power, stated plainly.** With N = 8 per arm, 0/8 vs 1/8 cannot distinguish "no effect"
   from "a small effect"; the rule of three bounds the real arm's hit rate at **≤ 37.5 %**
   (95 %). What the experiment *does* establish is the absence of the large effect the
   hypothesis needed: the reachable target is hit 2/2 by the oracle edits and 0/8 when the
   model chooses the edits itself, with the real evidence in front of it.
5. **The model did respond to the evidence — just not usefully.** Several real repairs
   activated dense terms at ~100 % during training and `real/L2_cand_00` even produced one
   in-training success termination, but at fresh seeds every one of them is 0/60 with
   `dock_entered` = 0.00.

### What follows for the project

Per the pre-registered interpretation, E1's failure means the negative result on this
environment is **capability-bounded, not evidence-bounded**: giving the loop the component
activation evidence it already has (and which `SESSION_STATE.md` §3g showed EUREKA's
reflection already carries) does not produce the repair. The pre-registered consequence is
that *adding* channels — including `HANDOFF_QUESTIONS.md` P5's trajectory-evidence extension
with its permuted-table control — is predicted to be **worthless on this environment**,
because the channel that already exists is sufficient in information content and still does
not work.

The one thing the experiment cannot rule out, and the natural next measurement: the real
edit was reachable *by me* reading the same evidence, so a loop with **iteration** (repair →
retrain → reflect again) or a **stronger operator** may still close it. This study tested one
round with one operator.

