# Pre-registration: does the evidence channel that ALREADY EXISTS let the model repair a reward?

Written 21:05, before any repair candidate was generated.

## Why this experiment is only possible now

Every LLM reward produced in this project scores 0/60, so **any** repair loop scored zero on
every arm and the method comparison was degenerate — there was no reachable target to find.
`runs/env_007/REPAIR_TEST_FINDINGS.md` changes that: two independent LLM candidates reach
**23/60 = 38.3 %** after exactly two localized edits (approach coefficient ×50; the one-off
terminal payoff converted into a dense per-step stream), and the two failure layers are
measured. So for the first time a repair arm *can* score above zero, and the question
"does the channel carry the information needed?" has a non-vacuous answer.

Note the deliberate framing: this is **not** a test of whether the model can reproduce *my*
two edits. The channel is the reflection report the pipeline already produces
(`build_reward_reflection`), and the question is whether that report — real vs sham — moves
the outcome. The edit is the model's to choose.

## Targets (8, both failure families)

The eight ladder candidates: `L2/{00,05,04,14}` (weak, gated advantage: `A` = −0.0004 …
−0.259) and `L0/{02,11,13,00}` (dense terms paying for the wrong behaviour: `A` up to +0.394,
never succeeds). One repair per target per arm.

## Channel

`build_reward_reflection(train_dir)` on each target's **1.2M** training summary. That report
contains, and the sham does not preserve:

* the task score, the training return curve, and the termination breakdown;
* a **per-component `episode_sum_mean` table over training windows**;
* a **per-component `active_rate` table over training windows** — i.e. exactly the
  activation evidence `SESSION_STATE.md` §3g established is *already present in EUREKA's
  reflection*, despite that function's stale docstring claiming otherwise;
* per-component `mean / abs_mean / min / max` over all training episodes.

**Sham construction (fixed):** within every per-component table, the **rows are shuffled and
then the values within each column are shuffled independently**, preserving the table's shape
and each column's marginal distribution while destroying both the name↔value correspondence
and the magnitude ordering. The task-score block and the training-return curve are left
identical in both arms, so the treatment is *specifically* the component-level evidence.

## Operator and protocol (identical in both arms)

`materialise_reward(mode="edit")` → `generate_edit` with `prompts/eureka_02_reward_edit.md`
(the config default), `deepseek-flash`, `temperature=0.7`, `max_tokens=16384`,
`DEEPSEEK_THINKING=disabled`, `max_validation_retries=3`; context
`runs/env_007/terminal_rule_pilot/seed_0/context`; config
`configs/env007_terminal_rule_pilot.yaml`. The key is read at runtime from line 19 of
`runs/env_001/ablation_eureka_feedback_v4/reproduction/run_ablation_score_only_v4.ps1` and
exported under the name the generation flow expects; it is never printed and never written to
a file. Nothing is handed to a subagent.

Training: `1.2M` steps, `n_envs=6`, clip 20, seed 0 — identical to every other run here.

**N = 8 repairs per arm** (one per target) → 16 trainings.

## Readout, on a NEW held-out block

Scored with `eval_pool.py` on fresh seeds **33000–33059** (60 episodes), used for the first
time here. Seeds 30000–30059 remain the historical headline block and 32000–32059 the block
used by the rung/repair studies; neither is reused for this experiment's headline number.
Secondary readouts, both training-free: the advantage-scale probe (`A`) and whether
`success_event` activation becomes positive.

## Bars and predictions, fixed now

| # | prediction |
|---|---|
| **E1** | the **real** arm produces at least one repair with **success > 0/60** on 33000–33059 |
| **E2** | the real arm's hit rate exceeds the sham arm's (one-sided Fisher on hits/total) |
| **E3** | sham arm hit rate is ≈ 0 (a model given shuffled component evidence has no more signal than one-shot generation, which is 0/60) |
| **E4** | among real-arm repairs, successes are accompanied by `A > 0` (the measured condition 1) |

**Two-stage rule (declared in advance):** if the first stage gives a one-sided Fisher
`0.05 <= p < 0.20` for E2, a second stage of 8 further repairs per arm is run and the stages
are reported both separately and pooled. If `p < 0.05` or `p >= 0.20`, the experiment stops
at one stage.

## Interpretation, fixed now

* **E1+E2+E3 hold** → the *existing* reflection channel carries the information needed to
  reach a repair; the bottleneck is one-shot generation, not the evidence available to the
  loop. This is the "evidence asymmetry" claim in its cleanest form: same model, same
  prompt, same N — only the channel's content differs, mounted on one side.
* **E1 fails** → even with the real component table in front of it, the model does not reach
  the repair. Then the project's negative result is **capability-bounded, not
  evidence-bounded**, and adding channels (P5's trajectory-evidence extension) is predicted
  to be worthless — which is a strong, decision-relevant negative.
* Either way, this says nothing about whether the *paper's* method (CREATE vs EUREKA) differs;
  it isolates the channel question, which is what P1–P5 asked for.

## Standing caveats

* The repair the model is asked for is now *known to exist* (two edits, measured), so a
  positive result is achievable by construction — that is the point, and it must not be
  reported as "the method now works on this environment".
* The 1.2M reflection is the richer of the two available budgets; a loop paying 0.6M would
  see mid-transit tables (`LADDER_FINDINGS.md` §3c). This is a design choice, recorded here.
* The key handling above is the only deviation from a literal reading of the constraint
  (which names `EUREKA_DEEPSEEK_API_KEY` while the on-disk line sets `DEEPSEEK_API_KEY`); the
  constraint's intent — read at runtime from that file, never retyped, never persisted — is
  respected and this note is the disclosure.
