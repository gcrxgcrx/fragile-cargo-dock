# CREATE v9 FINDINGS — CREATE reaches a delivery-level reward in round 1, and 8 of 10 rounds deliver

Protocol, predictions and declared confounds were fixed in `CREATE_V9_PREREGISTRATION.md`
before any round was scored. Raw artifacts: `runs/env_007/fragilecargo_create_v9/`
(`experiment_summary.md`, `seed_0/iter_XX/`, `eval_block41000.json`, `eval_block41000.md`),
launcher log `runs/env_007/fragilecargo_create_v9.log`.

## 1. The result

Ten rounds, seed 0, 3 M steps each (30 M environment steps — the same budget as the EUREKA
baseline and as the earlier CREATE run). Selection score is the pipeline's own in-training
evaluation (20 episodes, seeds 10000-10019). The fresh score is a held-out 60-episode block
**41000-41059**, previously unused, evaluated through the same code path as every other
fresh-seed number in this project (`eval_pool.py`).

| round | training selection score (20 eps) | **fresh 41000-41059** | dock | mean native return |
|---:|---:|---:|---:|---:|
| 1 | 279.630 (18/20) | **55/60** | 0.95 | 284.2 |
| 2 | 204.458 (13/20) | 25/60 | 0.88 | 133.5 |
| 3 | 279.630 (18/20) | **55/60** | 0.95 | 284.2 |
| 4 | 263.777 (17/20) | 51/60 | 0.88 | 263.9 |
| 5 | 309.545 (20/20) | **56/60** | 0.93 | 289.1 |
| 6 | 309.545 (20/20) | **56/60** | 0.93 | 289.1 |
| 7 | 294.207 (19/20) | **56/60** | 0.93 | 289.4 |
| 8 | 309.545 (20/20) | **56/60** | 0.93 | 289.1 |
| 9 | **309.773 (20/20)** | 53/60 | 0.90 | 274.1 |
| 10 | 53.867 (3/20) | 1/60 | 0.80 | 12.4 |

**Distribution over the ten rounds: n = 10, mean 46.4/60, sd 17.6, best 56/60, 8 of 10 rounds
at or above 50/60, 8 of 10 at or above 40/60.**

Reference points on comparable held-out blocks:

| reward / method | fresh-60 | source |
|---|---:|---|
| **CREATE v9 (this run), best round** | **56/60** | `eval_block41000.json` |
| EUREKA v9 + fixed spec, best of its full 10 candidates | 56/60 (gen 3) | `PIPELINE_CONTEXT_DIAGNOSIS.md` §4 |
| EUREKA v9 + fixed spec, generation 0 (4 candidates) | 53/60 | `fragilecargo_eureka_v9fix/eval_block39000.json` |
| single-shot v9 candidate `cand_02` | 58/60 | `V9_STRUCTURE_FINDINGS.md` (block 38000) |
| hand-written `probeD` | 45/60 | `ABLATION_FINDINGS.md` |
| **CREATE baseline (`fragilecargo_create_v2`)** | **0–1/60** | `RESEARCH_LOG.md` §3 |

## 2. What this establishes

1. **CREATE's first round is now a delivery-level reward.** Round 1 scored 279.63 in training
   (18/20 success terminations) and **55/60** on the fresh block. Prediction P2 from the
   pre-registration (`> 20` in round 1, strictly above the CREATE-v2 round-1 score of 4.504) is
   met with a wide margin: the baseline's *best* round over ten rounds was 17.694, and this run's
   *worst* of the first nine rounds is 204.458.
2. **The two changes are jointly sufficient for that, and neither was tested alone.** Round 1
   contains the spec fix and the structure block together (P1: the generated reward implements
   all nine native terms with the native coefficients). No arm in this run separates them.
3. **The reward CREATE writes has the native reward's economics.** Round 1's realised composition
   is `terminal_success` **95.97 %** of return (per-episode sum 270.0), `dock_enter` 1.78 %
   (5.0 once), `progress` 1.46 % (active 0.72), `approach_cargo` 0.62 % (active 0.99),
   `time_cost` −0.39, `action_cost` −0.06, `roughness` −0.03. The native reward puts ~96 % of its
   return in the terminal term (`RESEARCH_LOG.md` §14b); the single-shot v9 candidate put 89 %
   there. **This is the first CREATE round whose reward mass sits where the environment's own
   reward puts it**, and it is what the pre-v9 CREATE runs never achieved — `fragilecargo_create_v2`
   round 1's largest component was `crate_dock_alignment` at 80.8 % of magnitude, `dock_entered`
   0.00, 0/20 terminations.
4. **Rounds 5-8 hold at the ceiling.** Four rounds converge on a training score of 309.545
   (20/20 terminations) and a fresh score of 56/60, the best fresh result of this run and one
   better than EUREKA's generation-0 candidate on its own block. Round 9 set the run's best
   training score (309.773, 20/20) with a slightly lower fresh score (53/60) — the training
   score is a good but not perfect predictor of the fresh score.
   **All ten rounds produced byte-distinct executable reward code** (`code_signature` over all
   ten: no repeats), so the identical training scores are ten different implementations landing
   on the same behaviour, not a duplicated artifact being re-trained.
5. **The search is not monotone, and round 10 shows how it fails.** Round 10 scored 53.867
   (3/20) and 1/60 fresh *with dock 0.80*: the policy reliably reaches the dock and then fails to
   settle. That is the same signature as the pre-v9 families and the `probeD`-era "arrives but
   cannot stop" failure — an edit that kept reachability and lost settling. Round 2 (25/60,
   dock 0.88) is a milder version. So the lineage both climbs and falls; the drop rule
   (`stop_after_solved_drop`) that fired after round 2 would have ended the run before rounds 5-9
   produced the ceiling.

## 3. What this does not establish

* **n = 1 training seed.** Every number above is seed 0. The handoff (§7.1) records that the
  single-shot candidate `v9/cand_02` scored 58 / 35 / 0 across three training seeds, so a single
  seed is a draw. **The spread across rounds (sd 17.6) is not the same quantity as the spread
  across training seeds, and this run measures only the former.**
* **The comparison against `fragilecargo_create_v2` is confounded.** That run used prompt files
  that are no longer on disk (its recorded prompt contains a "必须落实的 failure modes" section the
  current `prompts/02_reward_generator_prompt.md` does not), and it ran on the v1 spec with a
  pre-v9 generator. The honest statement is "CREATE run with the fixed spec and the structure
  block does Y", not "the fix improved CREATE by X". A clean attribution would need the
  structure block and the spec fix as separate arms at equal budget.
* **EUREKA-vs-CREATE is still not a controlled method comparison.** The two flows differ in
  prompt shape, candidate-per-generation count, and stopping rule, and no EUREKA run has been
  made with `--seed 1`, so the method's own noise floor is unmeasured (handoff §6.2, still open).
  What can be said is narrower and still useful: at equal 30 M-step budgets on this environment,
  **CREATE delivered in its generation 0** (53/60 at round 1 on a different block, 55/60 here),
  which is the same "first generation is already enough" behaviour measured for EUREKA-v9 with
  the same information.
* **The structure block is not something CREATE extracted.** It is text taken from
  `prompts/eureka_01_initial_reward_v9.md`, given to CREATE by configuration. `V9_STRUCTURE_FINDINGS.md`
  §4 already records this limitation for EUREKA ("the candidate did not have to discover the
  structure"); it applies here unchanged.

## 4. Deviations from the pre-registered protocol, and why

The pre-registration said "10 rounds × 3 M, run to completion". Three deviations happened, and
the first two are the pipeline's own stop rules rather than mine:

| what | when | why | effect |
|---|---|---|---|
| stopped after round 2 | `stop_after_solved_drop_keep_best` | round 1 (279.63) set `solved_seen`; round 2 (204.46) fell below target 250 | resumed from round 3 with `--no-early-stop` |
| stopped after round 4 | `stop_solved_no_improvement_keep_best` | patience rule, patience 2 | resumed from round 5 with `--no-early-stop-all` |
| stopped after round 7 | `stop_solved_no_improvement_keep_best` **again** | `--no-early-stop-all` was inert: the override was placed *before* the config read, so `patience_after_solved = rounds` was immediately overwritten by the config's 2 | fixed (`tools/check_no_early_stop_flag.py`); resumed from round 8, which ran to `completed_all_rounds` |

Rounds 1-7 therefore ran with CREATE's own stopping rules armed (they fired twice), and rounds
8-10 ran with every adaptive stop disabled. **The scores are unaffected** — each round's score
comes from its own training run, and the rounds are seeded identically whether or not a stop rule
is armed — but a reader should know that the "completed_all_rounds" summary corresponds to rounds
8-10, not to the whole lineage.

Three further defects were found and fixed during the run; all are documented in the
pre-registration §5 with the measurements behind them, and all are independent of the result:

* truncated environment cards were accepted as valid (analyzer returned `finish_reason='length'`
  on ~half of calls; a 146-byte card was fed to the generator) — fixed with
  `llm.min_chars_env_card`;
* the subagent investigator never ran, in this run or in `fragilecargo_create_v2`'s ten rounds
  (`TypeError: run_investigator() got an unexpected keyword argument 'max_turns'`, swallowed by
  the caller) — fixed, regression check `tools/check_investigator_interface.py`. After the fix,
  **11 of this run's 12 subagent calls produced a research signal** (496-1505 chars); the one
  failure was a JSON parse failure, which the investigator does not retry (it is single-call by
  design), so that round's reflection ran without an investigation signal. The fix was applied
  before round 1, so all ten scored rounds had the stage live;
* on resume, `solved_seen` was reconstructed by substring-matching a line instead of reading the
  memory table's decision column, so it was always `False`; and `best_iter` was left unset on
  ties — both fixed.

## 5. Open, in the order I would take them

1. **Replicate rounds 5-9's reward at ≥ 2 more training seeds** on block 41000 before quoting
   56/60 as a property of a candidate. The run gives ten draws from the method; it gives one draw
   per candidate.
2. **Separate the two changes**: spec fix alone and structure block alone, at the same budget. The
   pre-registration predicted (P2) that the block is what matters, but this run cannot show it.
3. **Measure the method's own variance**: EUREKA with `--seed 1` (handoff §6.2) so the CREATE-vs-EUREKA
   comparison has a noise floor.
4. **The settling failure is still the open mechanism.** Round 10 (dock 0.80, 1/60) and round 2
   (dock 0.88, 25/60) fail in the same place: the policy reaches the dock and does not complete
   the 10-step settled predicate. `RESEARCH_LOG.md` §11 already isolates settling as the
   high-variance step; this run now provides eight rounds that *do* settle and two that do not,
   from one search lineage — a matched pair for studying what the successful rounds' rewards
   contain that the failing ones' do not.
5. **The edit operator now has a positive case.** `REPAIR_LOOP_FINDINGS.md` measured the operator
   as inert (real 0/8 vs sham 1/8) on the old information set. Rounds 2→5 here raise the score from
   204.458 to 309.545 and rounds 9→10 lose it; with the term table in the reflection context, the
   operator moves the reward in both directions. That is not the pre-registration's P4 ("at least
   one round raises the score by > 15 from a previous best below 250" — no round did, because the
   best was above 250 from round 1), so P4 is **not** met as written; what the run does show is
   that edits change behaviour substantially in either direction.
