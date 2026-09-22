# PIPELINE CONTEXT DIAGNOSIS — the missing dock geometry, and how much it actually cost

Supersedes the first version of this file, which claimed the wrong task spec made the pipeline
"fail permanently". That claim was based on generations 0-1 only. The run was left to finish,
and **it recovered on its own**: see §4. The corrected reading is below.

## 1. What was found (still correct)

The configs the pipeline used inherited

    inputs.task_spec_path = envs/env_007/task_spec_anonymized.yaml      (6,821 chars)

which does **not** contain the dock geometry, while the pilot path used

    envs/env_007/task_spec_anonymized_v2.yaml                          (7,373 chars)

which adds the `GEOMETRY OF 'FULLY INSIDE THE DOCK'` block (0.84 m bay, 0.60 m crate,
`|obs[12]| <= 0.024`, `|obs[13]| <= 0.030`). The v2 spec is a strict superset of v1: same 22
keys, one extra line.

Consequence in the analyzer's output (one API call, `run_01_environment_analyzer_md.py`):

| environment card | says about the dock tolerance |
|---|---|
| pipeline (v1 spec) | "**精确 dock 矩形尺寸、dock 朝向、crate 是否"完全进入"dock 不可直接验证**" - no numbers |
| pilot (v2 spec) | "可由 `obs[12]`, `obs[13]` 与几何阈值精确重建（0.024 / 0.030）" |
| regenerated with the fix (`configs/env007_fragilecargo_eureka_v9_fixedspec.yaml`) | "阈值：`|obs[12]| <= 0.024` 且 `|obs[13]| <= 0.030`，对应实际 `|x| <= 0.12 m`" |

And in the generated rewards:

| run | candidates that use the exact tolerance | selection score |
|---|---:|---|
| pipeline, v1 spec, generation 0 | **0 of 4** (six of all eight either omitted it or guessed 0.20/0.25) | 0.18-6.67, **0/20 terminated** |
| pipeline, v2 spec (fix), generation 0 | **2 of 2** | 172.7 / 0.04 |

## 2. Why a too-wide predicate is catastrophic for the *early* candidates

A reward that treats "within 0.20 normalised distance" as success fires its completion bonus and
its settled-state payoff in a region where the environment never terminates (it needs 0.024).
The policy settles into an attractor that pays forever and completes nothing. That is exactly the
measured signature: score stalls at 0.2-6.7, 0/20 success terminations, and the shaping terms are
the only thing being collected - the same signature as every failing LLM family since §3h.

## 3. The fix

`configs/env007_fragilecargo_eureka_v9_fixedspec.yaml` - one line:

    inputs:
      task_spec_path: envs/env_007/task_spec_anonymized_v2.yaml

Everything else inherits from `env007_fragilecargo_eureka_v9.yaml`. Verified: the regenerated
card contains the geometry, and both generation-0 candidates use the correct tolerance.

## 4. What it actually cost - the correction

The v1-spec run was left running to completion (4 generations, 10 candidates, 3 M steps each).
Its per-candidate scores:

| generation | candidates | scores |
|---|---|---|
| 0 | g00c00..03 | 3.90 / 0.18 / 6.67 / 2.08 (all 0/20) |
| 1 | g01c02, g01c03 | 3.53 / 3.61 (all 0/20) |
| 2 | g02c02, g02c03 | −1.78 / −0.82 |
| 3 | g03c02, g03c03 | −2.11 / **278.71 (18/20)** |

and on the shared fresh block **39000-39059**:

| candidate | origin | selection score | **fresh-60** |
|---|---|---:|---:|
| **`g03c03`** | **v1 spec (the "broken" one)**, generation 3 | 278.7 (18/20) | **56/60 = 93.3 %**, dock 0.93 |
| `cand_00` | v2 spec (fix), generation 0 | 172.7 (11/20) | 20/60, dock 0.50 |
| `cand_00` | pilot card, control | 203.8 (13/20) | 34/60, dock 0.77 |

**So the missing geometry did not make the pipeline incapable; it made the first two generations
worthless and pushed the recovery to the last generation.** The pipeline's own edit path - which
the repair-loop experiment had measured as inert - did climb out of the hole, guided by a live
score signal (and `g03c03` does not even use the exact tolerance: it found a working predicate by
another route).

## 5. Corrected consequences

1. **The earlier "0/20 vs 0/20, both methods degenerate" is partly explained**: on the prompt and
   spec actually used, generation 0 is all-but-guaranteed to fail (0.18-6.7), so a run that does
   not reach generation 3 has nothing. Both of the project's original API runs (CREATE 10 rounds,
   EUREKA 4 generations) did run to completion, so this alone does not explain their 0/20 - that
   remains open.
2. **The fix is still worth keeping**: it moves a working candidate from generation 3 to
   generation 0 (20/60 at 1.2 M in the two-candidate check), which is a large saving in the
   budget a search needs before it sees any signal.
3. **The headline measurement for the study is now this**: an LLM population search, with the
   geometry present, produced in **one generation of two candidates** a reward that delivers
   **20/60**; the unedited v9 single-shot candidates reach **55-58/60**; and the best candidate
   found by the full 10-candidate, 4-generation, 3 M-per-candidate run scores **56/60**.
4. **My error, recorded**: I asserted a permanent failure from generations 0-1. The run's own
   later generations falsified it. This is the third time in this session that an early sample
   produced an over-strong conclusion, and it is the reason the pre-registration rules in this
   repository insist on reading the whole trajectory before writing a verdict.
