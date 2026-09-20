# Pre-registration: v7 prompt — does naming the two measured conditions fix the ceiling?

Written before any v7 candidate was generated.

## The finding that motivates v7 (and why this is a prompt-level fix, not a new channel)

`runs/env_007/REPAIR_TEST_FINDINGS.md` measured what a working reward on this task needs:
(1) the candidate's **own approach term ~50× stronger**, so that acting beats idling, and
(2) a **dense per-step payoff while the instantaneous success predicate holds**. Two
independent LLM candidates reach **23/60 = 38.3 %** with exactly those two edits.

The scaffold in use (`prompts/eureka_01_initial_reward_v5.md`) **forbids edit (2)**:

* line 112: "完成奖励必须是一次性事件，不能在'成功状态'上每步给分（**违反即无效**）";
* self-check ④: call the reward 12 times on the same settled state and require the total to
  **stop growing** — i.e. force the model to rewrite any per-step settled payoff;
* its justification is a real measurement — a per-step settled payoff made a policy grind in
  the dock (43 steps, 30/60 episodes reaching 5–9 consecutive steps, 0 % delivery) — but the
  attribution was wrong. That measurement is `control_v1`, whose `A = −0.26`: the grinding came
  from the approach term being far too weak, so re-farming beat completing. Add the ×50
  approach term to a reward that has the stream and delivery appears (`r09`); the stream alone
  changes nothing (`r06`, `g01_stream` ≡ their copy controls).
* line 154's prescribed cure for "gets in but cannot stop" — a one-off dock-entry bonus — is
  measured not to work: `r04` has it, reaches `dock_entered` 0.97, and still scores 0/60.
* v5 also mandates a gentleness signal, and stacking gentleness on a candidate that already
  penalises closing speed **destroyed** behaviour (`r08`: dock 0.00; `r10`: 0/60).

**v7** = v5 with four auditable replacements (`make_prompt_v7.py`, diff in
`runs/env_007/v7_prompt.diff`; v5 untouched):

1. the absolute ban becomes a **conditional rule requiring both** a one-off completion event
   *and* a per-step settled payoff, with the strength ratio named as the thing that decides
   whether grinding happens;
2. the requirement list gains the mandatory per-step payoff and demotes the one-off dock-entry
   bonus explicitly ("it cannot replace the settled payoff: measured dock entry 0.97 with
   0/60 delivery");
3. self-check ④ is inverted (the total must grow linearly on a settled state) and a new
   **self-check ⑤** adds the advantage calibration: `R_push > R_idle` by at least the largest
   penalty's per-step magnitude, `R_settled > R_push`, and a warning against compressing all
   positive terms to a thousandth of the shaping magnitude — i.e. the measured `A ≤ 0` failure
   mode, stated as a design rule;
4. the gentleness clause keeps the signal but forbids stacking duplicates that swamp the
   completion payoff.

## Protocol

`pilot_generate_only.py`, `prompts/eureka_01_initial_reward_v7.md`, the **same** context
(`runs/env_007/terminal_rule_pilot/seed_0/context`), config
(`configs/env007_terminal_rule_pilot.yaml`), model (`deepseek-flash`) and `temperature=0.7` as
the L2 arm of the ladder, so the only difference is the prompt. **N = 8** candidates
(`runs/env_007/prompt_ladder_v7/`).

Training: 1.2M steps, `n_envs=6`, clip 20, seed 0 (identical to everything else).
Evaluation: fresh seeds **34000–34059** (60 episodes) — a block used for the first time here,
keeping 30000–30059 (headline), 32000–32059 (rung/repair) and 33000–33059 (channel) untouched.

## Baseline, fixed now

The v5-family measured rate is **0/16** at 1.2M with 60-episode fresh scoring:
the eight ladder candidates (4 `L2` + 4 `L0`) and the eight real-arm channel repairs. All had
`dock_entered` = 0.00 except `L0` arms which never succeeded either.

## Predictions

| # | prediction |
|---|---|
| **V7-1** | at least one v7 candidate scores **> 0/60** (baseline 0/16) |
| **V7-2** | the generated code contains a per-step settled payoff (training-free check on the reward files: a term that grows over repeated calls on a settled state — the check that v5 *forbade*) |
| **V7-3** | the advantage readout `A > 0` for at least half the candidates (the v5 family was `A ≤ 0` for 7 of 8) |
| **V7-4** | any successful candidate has `dock_entered > 0` (all eight channel repairs had 0.00) |

**Two-stage rule, declared in advance:** if V7-1 holds with one-sided Fisher
`0.05 <= p < 0.20` against 0/16, a second stage of 8 further v7 candidates is run and the
stages are reported separately and pooled. If `p < 0.05` or `p >= 0.20`, the experiment stops.

## Interpretation, fixed now

* **V7-1 holds** → the ceiling was reachable by a **prompt-level** change that names two
  measured conditions; EUREKA/CREATE re-runs with this prompt become worth their cost, and the
  method comparison stops being degenerate. It does **not** mean the search finds the reward
  unaided — the prompt does the work, and that must be reported as such.
* **V7-1 fails** (0/8 again) → naming both conditions explicitly is *still* not enough for this
  operator. Combined with real-vs-sham 0/8 vs 1/8 (full evidence, no prompt rule), the
  conclusion hardens: on this environment the ceiling is the operator's ability to implement
  the conditions it is told, not the information it has. That is the strongest form of the
  project's negative result and should be written as such.

## STATUS: COMPLETE — resumed 01:15, scored 01:45

### OUTCOME (fresh seeds 34000–34059, 60 episodes)

| candidate | `A` (training-free) | fresh-60 | dock | mean return |
|---|---:|---:|---:|---:|
| `cand_01` | **+4.282** | **3/60 = 5.0 %** | **0.18** | 18.36 |
| `cand_02` | +0.302 | 0/60 | 0.00 | +3.03 |
| `cand_03` | +0.059 | 0/60 | 0.00 | +0.09 |
| `cand_05` | +3.109 | 0/60 | 0.00 | +0.20 |
| `cand_00` | −0.194 | 0/60 | 0.00 | −0.01 |
| `cand_06` | −0.190 | 0/60 | 0.00 | −0.37 |
| `cand_07` | −0.685 | 0/60 | 0.00 | −1.90 |
| `cand_04` | −1.315 | 0/60 | 0.00 | −3.59 |

**hits 1/8; one-sided Fisher vs the 0/16 v5-family baseline = 0.3333.**

* **V7-1 HOLDS** (≥ 1 success) and **V7-4 HOLDS** (the success has `dock_entered` 0.18 > 0).
* **Outside the pre-declared stage-2 band `[0.05, 0.20)` → the experiment stops at one stage**,
  as pre-registered. The single hit is **not statistically distinguishable** from the
  v5-family baseline.
* My pre-registered prior ("at most 1–3 hits, and the four `A < 0` candidates stay immobile")
  is confirmed at its low end: the three `A < 0` candidates measured did stay at 0/60 with
  `dock_entered` 0.00.

### What actually moved, and what did not

**Moved (all measured):**

* the mechanical effect of the prompt change is complete: **8/8 candidates now pay per step on
  a settled state** (v5 family: 0/2; the working hand-written `probeD` is a stream);
* **`cand_01` is the first unedited LLM candidate in this project that reaches the dock**:
  `dock_entered` **0.18**, where *every* previous LLM candidate (35+, across v1–v6 and the
  channel repairs) measured **0.00**. The v5 family at 1.2M is 0/16 with dock 0.00 everywhere;
* in-training mean returns rose to the best ever seen for this generator (+3.05 / +3.30 for
  `cand_01`/`cand_02`; the v5 family spans −67…+1.55, and its best, +1.53, is the
  reward-hacking `L0/cand_02`);
* within the v7 family the in-training return *did* select the docking candidate (`cand_01` was
  second by in-training return and is the only hit) — unlike the v5 family, where the best
  in-training return was the reward hacker. Weak, one observation, but the right direction.

**Did not move:**

* the magnitude. **3/60 = 5.0 % is *below* the project's historical best-ever LLM observation
  (5/60 = 8.3 %, itself non-significant: two-sided p = 0.0573) and 7.7× below the oracle
  two-edit repair (23/60 = 38.3 %)**;
* 7 of 8 candidates still never enter the dock, i.e. the "immobile" failure mode persists for
  most of the family — and those are exactly the candidates whose measured `A ≤ 0`.

### Honest reading

The prompt fix is **real but insufficient**. It removed a self-imposed prohibition that had no
business being there (and whose justification was a mis-attributed measurement), and that
produced two qualitative firsts — the stream form in every candidate, and a docking candidate.
But the honest hit rate, 1/8 with 3/60 success, sits inside the project's noise floor, so this
**does not** license running EUREKA/CREATE as a headline experiment, and it does **not** show
that CREATE differs from EUREKA (the change is in the shared prompt).

The one lever the data now points at: the family splits cleanly on `A`. Four candidates have
`A > 0` (one is the hit); four have `A < 0` and all four measured were `dock_entered` 0.00.
The v7 prompt *asks* for the advantage calibration (self-check ⑤) but nothing enforces it. Making
that check **mechanical** in the pipeline's existing `validate_code` path — reject a candidate
whose reported `R_idle / R_push / R_settled` are not strictly increasing, or whose measured
`A ≤ 0` on the scripted library — is the cheapest next test, and unlike the earlier
"mechanical gate" (`SESSION_STATE.md` §4 error #5, which checked *structure* and had no
predictive power) this one checks the two conditions that were *measured* to be necessary.

---

## Resume instructions (kept for the record)

Everything below the prompt change itself is done except the honest scoring.

**Done**

* `prompts/eureka_01_initial_reward_v7.md` created by `make_prompt_v7.py` (4 auditable
  replacements of v5; v5 untouched; diff in `runs/env_007/v7_prompt.diff`).
* 8 candidates generated with v7 (`runs/env_007/prompt_ladder_v7/cand_00..07`), all valid, same
  context / model / temperature as the L2 ladder arm.
* **V7-2 HOLDS: 8/8 v7 candidates pay every step on a settled state** (median of repeat calls
  2–12 = 20.0, 9–11 of 11 non-zero), while both v5-family L2 candidates are one-off
  (median 0.000, 1 of 11 non-zero) and the working hand-written `probeD` is a stream (19.974).
  Tool: `check_settled_stream.py`.
* **V7-3 HOLDS at the boundary: 4/8 v7 candidates have `A > 0`** (+4.28, +3.11, +0.30, +0.06)
  versus 1/8 in the v5 family. Raw: `runs/env_007/advantage_probe_v7.json`.
* 3 of 8 candidates trained to completion (1.2M) — `cand_00`, `cand_01`, `cand_02`. In-training
  read only (20 episodes, seeds 10000–10019): **0 success terminations in all three**, but
  mean native returns **+0.08 / +3.05 / +3.30**, the highest of any LLM candidate measured in
  this project (the v5 family spans −67…+1.55, and the best of those, +1.53, is the
  reward-hacking `L0/cand_02`). Delivery-level returns are 124–310, so these are still ~40×
  short.

**Pending**

1. Train `cand_03`–`cand_07` (5 candidates). `cand_03` was killed mid-run; its partial
   `monitor/` has been removed. The paused queue had not started 04–07.
2. One honest evaluation of all 8 on fresh seeds **34000–34059** (60 episodes) into a single
   JSON, then the pre-registered verdicts V7-1 / V7-4 and one-sided Fisher against the 0/16
   v5-family baseline.

**Resume commands** (from the repo root; train only the missing five — `train_queue.ps1`
re-trains a candidate that already has a summary, so build the spec without 00–02):

```powershell
$g="D:\Code\python\research\form_github\expert-reward-agent"
$jobs=@()
foreach ($i in 3..7) { $n="cand_0$i"
  $jobs += [pscustomobject]@{ name="v7_$n"; reward="runs/env_007/prompt_ladder_v7/$n/reward_v1.py"
    save_dir="runs/env_007/prompt_ladder_v7/$n/training"; total_timesteps=1200000; seed=0; eval_episodes=20 } }
$jobs | ConvertTo-Json -Depth 4 | Set-Content "$g\runs\env_007\v7_resume_spec.json" -Encoding utf8

& $g\runs\env_007\train_queue.ps1 -Spec runs/env_007/v7_resume_spec.json -MaxParallel 4

# then score all 8 (adjust the python path to the project interpreter)
& D:\Code\python\research\llm_env_310\Scripts\python.exe $g\eval_pool.py --episodes 60 `
  --seed-offset 34000 --out $g\runs\env_007\prompt_ladder_v7\eval_block34000.json `
  (0..7 | ForEach-Object { "runs/env_007/prompt_ladder_v7/cand_0$_/training" })
```

**Resource note for the resume.** This was paused because free **commit** had fallen to
1.48 GB with 372 processes and a running game (VALORANT, 3.2 GB) on the machine; the queue's
`-MinFreeCommitGB 3.0` guard had already stopped it from starting a second wave. Do not resume
while a game is running: the failure mode is worker death during `import torch`
(`WinError 1455`) with the parent hanging at 0 CPU, which is what killed two earlier attempts
(`runs/env_007/ladder_train/README.md`). Kill the queue job with `job_kill` rather than
leaving it sleeping on the guard if the machine needs to be freed.

**Prior, recorded before the score exists** (so it can be checked): with V7-2 full but V7-3
only half, I expect **at most 1–3 of 8 hits**, and the four candidates with `A < 0`
(`cand_00`, `cand_04`, `cand_06`, `cand_07`) are expected to stay immobile.

---

## Standing caveats

* v7 is authored by me from measurements on this environment; it is a *scaffold* change, the
  same category as v2–v6, and it must be reported as part of the prompt lineage rather than as
  a search result.
* N = 8 per stage is small; the pre-declared two-stage rule is the mitigation and the power
  limitation must be stated with any positive.
* A v7 candidate reaching 38 % would not by itself show CREATE > EUREKA; it changes the
  *prompt*, which is shared by both.
