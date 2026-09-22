# SESSION HANDOFF — context-reset package (compressed)

**Purpose.** Read this after a context reset. It supersedes `NEXT_SESSION.md` for the
immediate next step, and it is written to be *short enough to read first*. Everything below is
measured; the raw artifacts are named next to each number.

**Repo**: `D:\Code\python\research\form_github\expert-reward-agent`
**Interpreter**: `D:\Code\python\research\llm_env_310\Scripts\python.exe` (never bare `python`)
**Fuller history**: `SESSION_STATE.md` (physics, harness, all earlier measurements),
`runs/env_007/RESEARCH_LOG.md` (baseline → v7 index),
`docs/external_reviews/fragilecargo_status_and_open_problems.md` (status + 3 addenda for an
external reviewer), `runs/env_007/V9_STRUCTURE_FINDINGS.md`.

---

## 1. The headline: an LLM-written reward now matches the environment's own reward

| reward | success on a 60-episode fresh block | source |
|---|---:|---|
| **`prompt_ladder_v9/cand_02`** | **58/60 = 96.7 %**, dock 0.98 | single-shot generation |
| `prompt_ladder_v9/cand_00` | **55/60 = 91.7 %**, dock 0.92 | single-shot generation |
| native reward through the wrapper (4 seeds) | 52 / 47 / 56 / 54 (mean 52.3, sd 3.9) | `passthrough_probe/` |
| hand-written `probeD` | 45/60 | `ablation_probe/probeD/` |
| **best unedited LLM candidate before this session** | **4/60** | `prompt_ladder_v8/cand_01` |
| every other LLM reward of the project | 0–5/60 | — |

`cand_02`'s mean native return is **299.44** against the native reward's canonical score
**299.65**; its max-consecutive-stable-step histogram is `{0 steps: 2, 10 steps: 58}`.

**Seed replication** (same block 38000-38059): `cand_00` = **55 / 54 / 52** (mean 53.7,
**sd 1.5**) — as stable as the native reward; `cand_02` = **58 / 35 / 0** (mean 31.0, sd 29.2).
Family distribution over all 8 candidates: **58 / 55 / 25 / 6 / 0 / 0 / 0 / 0**
(mean 18.0, sd 25.2, 3 of 8 above 20/60).

## 2. What v9 is, and why it worked where v7/v8 failed

`prompts/eureka_01_initial_reward_v9.md` = v8 **+ a block that states the reward's structure**:
the environment's nine native terms with their semantics and weights, how each is recovered from
`obs`/`action`, and a precedence rule. Every privileged channel stays closed (`info` forbidden;
the impulse term must be replaced by an observation proxy). Generator: `make_prompt_v9.py`,
diff `runs/env_007/v9_prompt.diff`, pre-registration + results
`runs/env_007/V9_STRUCTURE_PREREGISTRATION.md` / `V9_STRUCTURE_FINDINGS.md`.

**The single input that mattered was information about the reward's structure** — not the
operator, the evidence channel, the search, the selector, or the coefficient scale (all four of
those were falsified as the cause during this session: see §5).

## 3. The pipeline had a real, separate defect — now fixed, and its cost is known

`configs/...` used `inputs.task_spec_path = envs/env_007/task_spec_anonymized.yaml`, which does
**not** contain the dock geometry; the pilot path used `..._v2.yaml`, which adds
`GEOMETRY OF 'FULLY INSIDE THE DOCK'` (0.84 m bay, 0.60 m crate → `|obs[12]|<=0.024`,
`|obs[13]|<=0.030`). Consequence: the pipeline's self-generated environment card says the dock
dimensions are **"not directly verifiable"**, and of its eight generated candidates **none used
the correct tolerance** (six omitted it, two guessed 0.20/0.25 — 7-8x too wide). A reward whose
success predicate is that wide fires its completion bonus in states the environment never
terminates in, so it farms its shaping terms: score 0.18-6.67, **0/20 success terminations**.

* **Fix**: `configs/env007_fragilecargo_eureka_v9_fixedspec.yaml` (one line: point at `_v2.yaml`).
  Verified: the regenerated card contains the geometry and the generated candidates use it.
* **Cost, measured, and NOT what I first claimed**: the un-fixed run **recovered on its own** at
  generation 3 — `g03c03` scores 278.7 (18/20) and **56/60 on the fresh block 39000-39059**. The
  defect made generations 0-2 worthless, not the run.
* Write-up with the correction: `runs/env_007/PIPELINE_CONTEXT_DIAGNOSIS.md`.

---

## 3b. CREATE has now been run with both fixes — round 1 is already delivery-level (2026-09-22)

`fragilecargo_create_v9` = CREATE (10 rounds × 3 M, seed 0) with the **fixed spec** *and* the
**v9 structure block** injected into the generator's and the reflection agent's prompts
(`configs/env007_fragilecargo_create_v9.yaml`; write-up `runs/env_007/CREATE_V9_FINDINGS.md`,
pre-registration `runs/env_007/CREATE_V9_PREREGISTRATION.md`). This completes §6 step 3 below.

| round | training (20 eps) | fresh block **41000-41059** |
|---:|---:|---:|
| 1 | 279.630 (18/20) | **55/60** |
| 2 | 204.458 | 25/60 |
| 3 | 279.630 | **55/60** |
| 4 | 263.777 | 51/60 |
| 5 | 309.545 (20/20) | **56/60** |
| 6 | 309.545 (20/20) | **56/60** |
| 7 | 294.207 | **56/60** |
| 8 | 309.545 (20/20) | **56/60** |
| 9 | **309.773 (20/20)** | 53/60 |
| 10 | 53.867 (3/20) | 1/60 |

**n = 10 rounds, mean 46.4/60, sd 17.6, best 56/60, 8 of 10 rounds ≥ 50/60** — against the
CREATE baseline's best of 17.694 in training and 0-1/60 fresh. Round 1's reward puts **95.97 %**
of its return in `terminal_success` (the native reward puts ~96 % there); the baseline's round 1
put 80.8 % of its mass in `crate_dock_alignment` and never docked. Round 10 is the failure worth
studying: dock 0.80, 1/60 — it arrives and cannot settle.

Three pipeline defects were found and fixed while running this; **all three also affected the
earlier runs**, so they are worth knowing regardless of CREATE:

* the environment analyzer returns a **truncated card on ~half of all calls** on this environment
  (`finish_reason='length'`; 146 B / 4.6 KB / 9.1 KB against a normal 14-20 KB) and the client only
  retried *empty* responses, so half-written cards were fed to the generator. Now guarded by
  `llm.min_chars_env_card` in `configs/env007_fragilecargo_eureka.yaml`. **Any earlier argument of
  the form "the card said X" should first check that card's length**;
* `run_investigator()` did not accept the config's `max_turns`, so **every reflection call raised
  TypeError and the whole subagent-investigation stage was silently skipped** — in this run's
  round 2 and in all ten rounds of `fragilecargo_create_v2` (that run's logs contain no
  `Subagent:` line at all). Fixed; regression check `tools/check_investigator_interface.py`;
* on resume, `solved_seen` was reconstructed with a substring test that can never match
  (`target_solved_new_best` contains no bare `target_solved` token) and `best_iter` was left unset
  on ties, so a resumed lineage treated an already-solved target as unsolved. Fixed in
  `pipeline/run_iterative_experiment.py`.

Protocol note: the lineage stopped itself twice (CREATE's own `stop_after_solved_drop` and
patience rules, after rounds 2 and 4) and a third stop after round 7 exposed that the new
`--no-early-stop-all` override had been placed before the config read and was inert. Rounds 8-10
ran with adaptive stops disabled; scores are unaffected. **Round 1 was already solved, so the drop
rule would have kept only a 55/60 candidate and thrown away the four 56/60 rounds.**

## 4. The EUREKA run with the fixed spec — RESULT (stopped after generation 0 on the user's call)

`fragilecargo_eureka_v9fix` = EUREKA (4 candidates per generation, 3 M steps each) with the
**fixed spec** and the **v9 prompt**. It was stopped after generation 0 because that generation
already produced a delivery-level candidate; the trained generation-0 models were then scored on
the shared fresh block **39000-39059**.

| candidate (gen_00) | in-training selection score (20 eps) | **fresh-60** | dock | mean return |
|---|---:|---:|---:|---:|
| **`g00c02`** | **294.18 (19/20 terminated)** | **53/60 = 88.3 %** | **0.92** | **273.8** |
| `g00c00` | 23.66 (1/20) | 2/60 | 0.87 | 18.2 |
| `g00c03` | 8.31 (0/20) | 1/60 | 0.90 | 13.3 |
| `g00c01` | 1.69 (0/20) | 0/60 | 0.00 | 2.0 |

Raw: `runs/env_007/fragilecargo_eureka_v9fix/eval_block39000.json`. The run directory holds the
four trained models, their rewards and their training summaries; generations 1-3 were never
trained, so `eureka_summary.md` does not exist for this prefix (expected).

### The three-way comparison that this completes (all on block 39000-39059)

| configuration | generation that first reached delivery | best fresh-60 | generation-0 scores |
|---|---|---:|---|
| **v9 prompt + FIXED spec** | **generation 0** | **53/60** | 1.7 / 8.3 / 23.7 / **294.2** |
| v9 prompt + broken spec | generation 3 (its 10th candidate) | 56/60 | 0.2 / 2.1 / 3.9 / 6.7 |
| v9 prompt + pilot context (control) | generation 0 | 34/60 | 4.8 / 203.8 |

**Readings, fixed:**
1. the fix moves a delivery-level candidate from the **4th generation to the 1st**: ~12 M steps
   instead of ~30 M steps to the same place;
2. generation-0 hit rate is **1 of 4** (294 / 24 / 8 / 2), the same order as the single-shot v9
   family's **3 of 8** — i.e. the pipeline's generation is now as effective as the one-shot path;
3. therefore the two defects act on different things: **the prompt (v9) sets the ceiling**
   (4/60 → 58/60), **the spec fix sets how soon the search sees signal** (gen 4 → gen 1).

### First things to do after the reset

```powershell
$py = "D:\Code\python\research\llm_env_310\Scripts\python.exe"
# 1) the result above, verbatim
Get-Content runs\env_007\fragilecargo_eureka_v9fix\eval_block39000.json -Raw
# 2) BEFORE quoting 53/60 as a property of the candidate: replicate it on >= 2 more training seeds
#    (the sibling v9 candidate cand_02 scored 58 / 35 / 0 across three seeds)
# 3) then build the noise floor for the method comparison: rerun EUREKA with --seed 1
```

## 5. What was falsified this session (do not re-propose without new data)

| hypothesis | how it died |
|---|---|
| "no training-free statistic can *select*" | confirmed five times over: structure checks, trajectory ordering, 0.6/1.0 M rungs, per-step advantage `A`, shaping:terminal ratio, hover wage, dock-dwell, dock-pay share — ρ ≤ 0.41 against success, bars were 0.60 |
| "the missing ingredient is the term's magnitude" | scaling the closing-speed coefficient 15x and 50x took docking 0.70 → **0.00** (immobility): `CALIBRATION_TEST_PREREGISTRATION.md` |
| "the isolation effect is real at the seed level" | 4 paired seeds: +45 / −22 / −54 / −22, 95 % CI [−79.5, +53.0], **p = 0.55**: `SEED_REPLICATION_FINDINGS.md` |
| "a 1.2 M run predicts a seed's class at 3 M" | `C1` seed 0: 45/60 at 1.2 M → **3/60** at 3 M |
| "success is dominated by seed variance, not the reward" | true **only for fragile rewards**; the native reward is 52/47/56/54 and `v9_cand_00` is 55/54/52 — a correct-structure reward is not a lottery |
| "the pipeline is broken/incapable" | the un-fixed run's generation 3 reached 56/60 |
| "the edit operator is inert" | this session produced two counterexamples: `g03c03` (+275 in one edit) and the v9 fixedspec verification (172.7 at generation 0) |

## 6. Candidate next steps, in the order I would take them

0. **The in-flight run is finished and scored** (§4): EUREKA with the fixed spec reached
   **53/60 on a fresh block using only the four candidates of generation 0**. Everything below is
   now about turning that into a defensible claim, not about finding a working reward.
1. **Replicate `g00c02` on ≥ 2 more training seeds** on the same block before quoting 53/60. The
   sibling single-shot candidate `v9/cand_02` scored 58 / 35 / 0 across three seeds, so a single
   seed is a draw, not a property.
2. **Measure the method's own variance**: rerun EUREKA with `--seed 1` (≈1.5 h). Without this the
   EUREKA-vs-CREATE comparison has no noise floor — two runs of one method can differ by more
   than two methods do.
3. ~~**Only then run CREATE** with the fixed spec~~ — **done** (§3b): `fragilecargo_create_v9`,
   10 rounds × 3 M, best 56/60 fresh, round 1 already 55/60. CREATE also received the v9
   structure block, so CREATE and EUREKA were run with the same reward information.
4. Report **distributions and cross-run variance**, not best-of-N: "fraction of candidates that
   deliver" is the quantity this environment's variance lets us compare. CREATE's distribution is
   now the first one measured: 8 of 10 rounds ≥ 50/60, mean 46.4, sd 17.6.
5. **Next, in order**: (a) replicate one ceiling round (5-8) at ≥ 2 more training seeds before
   quoting 56/60 as a property of a candidate; (b) separate the two changes with a spec-fix-only
   arm — this run cannot attribute the result to the spec fix versus the structure block;
   (c) EUREKA at `--seed 1` for the method noise floor (item 2 above, still open).

## 7. Hard constraints (unchanged, still binding)

* API keys are read **from `D:\Code\python\research\DSapi.txt` at runtime** and must never be
  printed or written into the repo. (`EUREKA_DEEPSEEK_API_KEY` for EUREKA flow,
  `DEEPSEEK_API_KEY` for CREATE; `DEEPSEEK_THINKING=disabled`; model `deepseek-flash` only.)
* **Never launch more than ~4 trainings at once** — the binding resource is the Windows commit
  limit; workers die inside `import torch` and the parent hangs at 0 CPU (`WinError 1455`).
  Run them detached (`subprocess`/`Start-Process`) so an interrupted shell does not kill them.
* Held-out evaluation blocks consumed so far: 30000, 32000, 33000, 34000, 35000, 36000, 37000,
  38000, 39000, **41000** (CREATE v9's ten rounds, `runs/env_007/fragilecargo_create_v9/eval_block41000.json`).
  **40000 is unused** — the v9fix monitor was written to score the best candidate there, but its
  `eval_best.log` is empty and no `eval_best_block40000.json` was produced, so the block was never
  actually run. **40000 and 42000+ are available.**
* A `git reset --hard` in this repo **deletes untracked files** (it cost ~2,750 run artifacts
  earlier this session; they were restored from the pushed clone). Back up before any reset.

## 8. Integrity note

The working repo's `master` is at `b53b2071` (29 commits ahead of `origin/master`, unchanged).
The external repo `https://github.com/gcrxgcrx/fragile-cargo-dock` holds the curated research
record through v9's predecessor (`main` = `61b32478`); this session's v9 artifacts are **not yet
pushed**. `runs/env_007/` on disk carries everything regardless.
