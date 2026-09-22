# CREATE v9 — pre-registration (protocol, predictions, declared asymmetries)

Written **before** any round of the run had been scored. At the time of writing, round 1 was
still training; the only things observed were mechanical:

* the round-1 environment card is **21,397 bytes** and contains the dock geometry
  (`|obs[12]| <= 0.024`, `|obs[13]| <= 0.030`) — the spec fix works on the CREATE path,
  which the earlier `PIPELINE_CONTEXT_DIAGNOSIS.md` had only verified on the EUREKA path;
* the round-1 generated reward contains **all nine native terms** with the native coefficients
  (`approach_cargo` +1.0/m, `progress` +1.0/m, `dock_enter` +5, `roughness` −0.02, `action_cost`
  −0.0005, `time_cost` −0.002, `hard_hit` −0.5, `terminal_success` +300,
  `terminal_failure` −100) — i.e. the structure injection reached the generator.

Neither of those is a performance result, and no score is quoted below.

Raw artifacts: `runs/env_007/fragilecargo_create_v9/`, launcher log
`runs/env_007/fragilecargo_create_v9.log`.

## 1. What is being run

| item | value |
|---|---|
| method | CREATE (`pipeline/run_iterative_experiment.py`): single lineage, structured diagnosis, semantic-localised edit, intervention memory, best-reward archive |
| config | `configs/env007_fragilecargo_create_v9.yaml` |
| rounds × budget | 10 × 3,000,000 environment steps = 30 M |
| evaluation during search | 20 episodes per round, seeds `10000-10019` |
| training seed | 0 (lineage seed; the reflection path is seeded `seed + 100·restart`) |
| env | `FragileCargoDock-v0` |
| model | `deepseek-flash`, `DEEPSEEK_THINKING=disabled`, temperature 0.0 (card) / 0.15 (generator) |
| launcher | `run_create_v9_full.py` (key read from `D:\Code\python\research\DSapi.txt` at runtime, never printed/written; retry + resume from the first round without `training_summary.json`) |

Matched comparison runs, all at 10 × 3 M:

* `fragilecargo_eureka_v9fix` — EUREKA, v9 prompt, fixed spec. **53/60** on block 39000-39059
  from its generation-0 candidate `g00c02` (`eval_block39000.json`).
* `fragilecargo_eureka_v9` — EUREKA, v9 prompt, v1 spec. Reached 56/60 at generation 3.
* `fragilecargo_create_v2` — CREATE, v1 spec, pre-v9 prompt. Best in-training score **17.694**,
  `0/20`-style success terminations throughout.

## 2. The two changes, and what each is for

1. **Spec fix.** `inputs.task_spec_path` → `envs/env_007/task_spec_anonymized_v2.yaml`, which adds
   the `GEOMETRY OF 'FULLY INSIDE THE DOCK'` block. Without it the analyzer cannot state what
   "fully inside the dock" means numerically, the generator guesses a tolerance 6-8x too wide,
   and the completion bonus fires in states the environment never terminates in. Measured cost on
   the EUREKA side: generations 0-2 worthless, recovery at generation 3.
2. **Structure injection.** `inputs.reward_structure_context_path` →
   `runs/env_007/CREATE_V9_STRUCTURE.md`, extracted **verbatim** from
   `prompts/eureka_01_initial_reward_v9.md` by `tools/extract_v9_structure_block.py`
   (source sha256 `bb3e15cd…`, block sha256 `f3fca2aa…`). It is written next to the card as
   `v9_structure_block.md` and appended to the generator's user prompt (after the generator
   prompt and the expert context, so its own precedence declaration governs the conflicts with
   the generic rules) and, under `context.include_reward_structure_in_reflection: true`, to the
   reflection agent's user prompt as section 6.5.

The block is the input that moved EUREKA from **4/60 to 58/60**; §5 of
`runs/env_007/V9_STRUCTURE_FINDINGS.md` records that the operator, the evidence channel, the
search and the coefficient scale were each falsified as the cause.

## 3. Predictions (fixed now)

* **P1 — the injection reaches the generator.** Round-1 reward implements all nine terms.
  *Already observed; not counted as a result.*
* **P2 — generation 0 is no longer worthless.** The CREATE-v2 baseline spent generations 0-2 at
  scores 4.50 / −102.84 / 2.13. With the card carrying the geometry and the generator carrying the
  term table, round 1 should score **> 20** (i.e. at least one success termination in 20 episodes)
  or at minimum strictly above the CREATE-v2 round-1 score of 4.504.
* **P3 — the ceiling, if reached, is reached late.** CREATE is a single lineage: unlike EUREKA's
  4-candidate generation, it gets one draw per round, so the *family* hit rate of the v9 prompt
  (~3/8 candidates ≥ 20/60) predicts roughly a 1-in-3 chance per round of a delivery-level
  candidate. Best-round score is therefore predicted to be **> 250 only if at least one of the ten
  rounds draws a good candidate**; I expect the run to be bimodal the same way the v9 family is.
* **P4 — the edit path is the open question, not generation.** The repair-loop study measured the
  operator as inert (real 0/8 vs sham 1/8, Fisher p = 1.0000) *on the old information set*. With
  the term table in the reflection context, the prediction is that at least one round **raises the
  score by > 15 (the configured `min_meaningful_improvement`) from a previous best below 250**.
  If no round does, the operator stays inert even when it is told what the native reward contains —
  which would be a stronger version of the repair-loop finding, not a failure of this run.
* **P5 — success/score decoupling is expected to persist.** Following §11 of `RESEARCH_LOG.md`,
  rounds are reported with `dock_entered` and success **separately**; a round with high docking and
  0 success is a settling failure, not a reachability failure.

## 4. Declared confounds and asymmetries (not hidden, not fixable by config)

1. **The prompt files changed since the CREATE baseline.** The `fragilecargo_create_v2` run recorded
   its generator prompt in `.../iter_01/generation/prompt_records/02_reward_generator.md`; that
   recorded text contains a "必须落实的 failure modes" section which the current
   `prompts/02_reward_generator_prompt.md` does **not** have. No config can restore the file that
   produced the baseline, so the comparison against `fragilecargo_create_v2` is confounded by the
   prompt revision *in addition to* the spec fix and the structure injection. Any statement of the
   form "the fix improved CREATE by X" is therefore **not supported** by this run; only "CREATE,
   run with the fixed spec and the structure block, did Y" is.
2. **Information asymmetry in the other direction.** EUREKA-v9's generator receives the structure
   block *inside its own prompt*; CREATE's receives it appended to the user message. The text is
   identical, the position differs. The position matters for instruction-following, and this run
   cannot separate the two.
3. **The CREATE generator prompt actively contradicts the block** on one point: it says
   `explicit_success_flag_available=false` ⇒ do not write `terminal_success_reward` as a core v1
   term, and budgets v1 at 2-4 components, whereas the block requires all nine terms. The block's
   precedence declaration resolves this in the text; the run measures whether it resolves it in
   behaviour (P1 is the mechanical check).
4. **One training seed.** `seed 0` only. Following §7.1 of the handoff, a single seed is a draw, not
   a property: `v9/cand_02` scored 58/35/0 across three seeds. No claim about CREATE's
   seed-stability can be made from this run.

## 5. Two pipeline defects found while starting this run, both fixed before the result

### 5a. Truncated environment cards were accepted

The first launch (02:19) wrote a **146-byte** environment card: the analyzer's completion stopped
mid-sentence and `DeepSeekClient.completion` only retried when the response was *empty*, so the
truncated text was accepted and handed to the reward generator.

Measured with `tools/diagnose_analyzer_truncation.py` (4 isolated calls, no run artifacts):

| attempt | chars | finish_reason | geometry present |
|---:|---:|---|---|
| 1 | 4,590 | `length` | yes |
| 2 | 13,828 | `stop` | yes |
| 3 | 11,879 | `length` | yes |
| 4 | 9,116 | `length` | **no** |

So the analyzer returns a truncated card on ~half of all calls on this environment. The fix is a
minimum-length guard (`llm.min_chars_env_card: 8000` in `configs/env007_fragilecargo_eureka.yaml`)
that makes such a response count as unusable and re-sample; re-tested, 3/3 calls then produced
complete cards (12.6 / 13.0 / 12.8 KB, all with geometry).

This matters beyond this run: **the EUREKA runs and the `fragilecargo_create_v2` baseline all ran
through the unguarded path.** Their reported selection scores are unaffected (those come from
training, not from the card), but any argument of the form "the card said X, therefore the model
had X available" in those runs needs the card checked for length first — a truncated card can be
missing whatever the argument relies on.

### 5b. The subagent investigator never ran — in this environment's previous CREATE run either

`run_reflection_agent.py` passes `max_turns=subagent_investigator.max_turns` to
`run_investigator()`, which did not accept that keyword. Every call therefore raised
`TypeError: run_investigator() got an unexpected keyword argument 'max_turns'`, the caller caught
it and continued without a research signal, and the whole subagent stage was silently skipped.
This was observed live on this run's round 2 (`Subagent: error (continuing without signal)`), and
it necessarily affected all ten rounds of `fragilecargo_create_v2` as well, since the config and
the call site are unchanged between them.

Fixed by accepting `max_turns` as a documented no-op (the investigator is single-call by design);
`tools/check_investigator_interface.py` is the regression check. After the fix, round 2 of the
first attempt logged `Subagent: 1 turns, signal=780 chars`.

### 5c. Restart

The run was stopped after round 1 of its first attempt and restarted from scratch, so that the
lineage reported here was never built on the truncated card and never ran a round with the
subagent stage dead. The first attempt's artifacts are kept for audit in
`runs/env_007/_attempt1_pre_fix_archive/` (log, round-1 reward, round-1 training summary).

Attempt 1, round 1, recorded for the record: score **3.538**, `terminated 0/20`,
`dock_entered 0.00`; component shares `progress` 0.56 (active 0.49), `approach_cargo` 0.28
(active 1.00), `time_cost` 0.13, `dock_enter` / `terminal_success` / `terminal_failure` all 0.000.
Its reward did implement all nine native terms — so the structure injection worked — but the policy
never reached the dock, and no term that pays for completing ever fired.

