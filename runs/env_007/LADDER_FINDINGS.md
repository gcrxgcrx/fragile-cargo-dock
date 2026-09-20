# LADDER FINDINGS — the pre-registered decision, and why every LLM candidate fails

Companion to `runs/env_007/LADDER_DECISION_PREREGISTRATION.md` (which fixed the decision
rule and the follow-up protocol *before* the outcomes existed) and to
`runs/env_007/ladder_train/README.md` (launch mechanics).

Scope reminder: experiments only. All paper-reporting issues stay in
`HANDOFF_QUESTIONS.md` §6.

---

## 1. The pre-registered decision, and its outcome

Eight 1.2M-step trainings (4 × `L2` oracle-scaffold, 4 × `L0` interface-only), chosen so
that *both* the structural checks and the outcome would have variance. Result
(`ladder_analysis.py`, fresh seeds 30000–30059, 60 episodes, clip 20):

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

`successes = [0,0,0,0,0,0,0,0]` → **outcome has no variance → every rho is undefined.**
By the pre-registered bar (`|rho| >= 0.60` on a defined rho), **no check qualifies**:
the branch taken is **Branch B — cheap probes are closed for selection.**

Secondary observation, explicitly *not* the pre-registered outcome: `event`, `second` and
`term_ratio` each reach rho = 0.524 against **mean return**, while success stays 0 for all
eight. Structural compliance moves the return and not the success. n = 8, p ≈ 0.18 — no
method conclusion may be drawn from it.

Two things this buys us:

* **11 of 16 L2 candidates passed all four structural checks at generation time
  (`SESSION_STATE.md` §3e), and the four trained here are 0/60.** Since the scaffold rules
  are what produce that compliance (event-writing goes 0–1/16 → 16/16 when they are added),
  this closes the "mechanical gate has predictive power" hypothesis on the *same* pipeline
  that produces the compliance — the earlier, weaker form of the refutation is error #5 in
  `SESSION_STATE.md` §4.
* The two independent batch-1 vs batch-2 logs agree with the ladder's own generation-time
  counts: `L0/cand_02` is the single L0 candidate that writes a terminal event, matching
  §3e's "L0 = 1/16".

## 2. Why every one of them fails: the decisive term is never active

`component_share_report.py` reads the per-component `active_rate` / `magnitude_share` of
the **final policy** out of each `training_summary.json`. Two groups, same table:

Hand-written arms at 1.2M (success from the fresh block, §4 below):

| arm | mean eval reward | episode terminations | dominant components (share) | fresh-60 |
|---|---:|---|---|---:|
| probeC | 309.59 | terminated 20/20 | `terminal_success` **96 %**, dock_enter 2 % | 59/60 |
| probeE | 264.06 | terminated 17/20 | `success_event` **96 %** (active_rate 0.0030), dock_enter 2 % | 46/60 |
| probeD | 218.08 | terminated 14/20 | `dock_settled_hold` **96 %**, crate_progress 3 % | 44/60 |
| probeA | 173.08 | terminated 11/20 | `dock_settled_hold` **97 %**, crate_progress 2 % | 26/60 |
| control v1 | 7.17 | terminated 0/20 | crate_progress 76 %, cart_approach 24 % — **no settled/terminal term** | 0/60 |

LLM candidates at 1.2M:

| cand | mean eval reward | episode terminations | dominant components (share) | `success_event` active_rate | fresh-60 |
|---|---:|---|---|---:|---:|
| L2/cand_00 | −1.23 | 0/20 | `dock_speed_penalty` **94 %** (a penalty), progress 6 % | **0.0000** | 0/60 |
| L2/cand_04 | −0.93 | 0/20 | `action_penalty` **100 %** | **0.0000** | 0/60 |
| L2/cand_05 | −1.33 | 0/20 | *(nothing active; mean reward exactly 0.0)* | **0.0000** | 0/60 |
| L2/cand_14 | −1.29 | 0/20 | `crate_dock_progress` **100 %** | **0.0000** | 0/60 |
| L0/cand_00 | −67.40 | terminated 13/20 | `out_of_bounds_penalty` 59 %, action_smoothness 41 % | — | 0/60 |
| L0/cand_02 | +1.53 | 0/20 | `crate_docking_quality` 94 % | — | 0/60 |
| L0/cand_11 | −2.26 | 0/20 | `crate_docking_quality` 72 %, `crate_dock_proximity` 27 % | — | 0/60 |
| L0/cand_13 | −1.86 | 0/20 | `crate_docking_quality` 100 % | — | 0/60 |

The pattern:

1. Every arm that works puts **~96–97 % of its reward mass into the terminal/settled
   success term**, whose `active_rate` can be tiny (probeE: 0.0030 — about 1 step in 330)
   yet whose per-activation magnitude dominates everything else.
2. Every LLM candidate puts **0 %** there. Its mass sits in penalties (`dock_speed_penalty`
   94 %, `action_penalty` 100 %) or in a progress term, and for `L2/cand_05` **no term is
   active at all** → the reward is identically 0 along the trajectory the policy actually
   produces → there is no gradient to climb.
3. `control v1`, the 0 % hand-written control, is the intermediate case: it has a real
   signal (mean reward 4.8e−2, larger than any L2 candidate) but no terminal term, and it
   also lands at 0/60.

**So the failure is not that the generated event terms are missing or mis-scaled in the
function — the checks certify `docked_raw` up to +30, `term_ratio = infinity`, and a
one-off event. It is that these terms are gated on states the policy never visits, so the
realised reward landscape is ~0 and nothing is learnable at this budget.** The structural
checks probe the *function* on states a scripted expert reaches; they never ask whether
the term is *active under the policy's own trajectory*.

This is the mechanism behind §3f ("ranking correctness ≠ learnability") and it explains
why the trajectory-ranking check also separates probes from a 0 % control perfectly: it
too evaluates the function on reachable-by-a-scripted-expert trajectories.

### 2.1 Two distinct failure modes, not one

The table separates two ways to fail, and they need different fixes:

* **Farmable dense term** (`L0/cand_02`, `L0/cand_11`, `L0/cand_13`, `control v1`).
  A positive term with `active_rate = 1.0` that pays continuously for *being near or in*
  the dock region. `L0/cand_02` collects **+311 per episode** — the largest realised reward
  of any candidate here, larger than every working probe — and still never satisfies the
  success predicate. The policy farms the term; the term does not require completion.
  This is the L0/interface-only signature: nothing in the prompt defines "done", so the
  generator writes a smooth quality score.
* **Unreachable sparse term** (`L2/cand_00`, `L2/cand_04`, `L2/cand_14`, `L2/cand_05`).
  A one-off event exists in the function (the scaffold forces it) but its realised
  `active_rate` is exactly **0** and everything else is a penalty or negligible
  (`+0.82` per episode for `L2/cand_14`, `+0.0047` for `L2/cand_00`, literally nothing for
  `L2/cand_05`). The landscape is flat at zero, so PPO has no gradient.

Only one combination works: a **positive, sparse-active (one-off) term that dominates the
mass *and* is actually reached** — which is precisely what the hand-written `probeA`–`probeE`
have (share 0.86–0.97, active_rate 0.0006–0.03, +60…+300 per episode).

This distinction is what §3c of the pre-registration turns into a frozen, testable
readout.

## 3. Consequence for selection (Branch B)

Selection among LLM candidates is **moot while the pool contains no positive candidate**:
a selector ranks; it cannot manufacture a signal that is absent. Branch B's rung study is
still the right next measurement — it tells us the resolution of the cheapest honest
instrument — and it was run: §5 reports it, and the answer is that no cheap rung works
either, so both sides of the selection question are now closed by measurement.

## 4. Bonus: independent replication of §3b on an unused seed block

The six hand-written arms were re-measured at 1.2M on a **new** block, seeds 32000–32059
(60 episodes), same code path as `eval_fresh_seeds.py` (`eval_pool.py`):

| arm | §3b (seeds 30000–30059) | this run (seeds 32000–32059) | delta |
|---|---:|---:|---:|
| control v1 | 0 % | 0 % | 0 |
| probeA | 40.0 % | 43.3 % | +3.3 |
| probeB | 45.0 % | 36.7 % | −8.3 |
| probeD | 73.3 % | **73.3 %** | 0 |
| probeE | 65.0 % | 76.7 % | **+11.7** |
| probeC | 98.3 % | **98.3 %** | 0 |

The **ordering is completely stable** (control < A ≈ B < D ≈ E < C) and the two extremes
reproduce to the decimal. But an individual value moves by up to **11.7 points** when only
the 60-episode seed block changes — about 2 binomial SE at p ≈ 0.5. Any selector study on
this environment must therefore compare candidates *within one block*, and must not treat
a few points of difference as signal. Raw data:
`runs/env_007/rung_06m/eval_1m2_block32000.json`.

## 5. Branch B rung study (0.6M as a selector) — **REJECTED**

Protocol: `LADDER_DECISION_PREREGISTRATION.md` §3 + §3b amendment. Six hand-written arms
whose 1.2M outcomes span 0 % → 98.3 %, retrained at **0.6M** with seeds {0,1,2}
(18 runs, `train_queue.ps1`), plus the eight ladder candidates at 0.6M (8 runs, seed 0).
All scored on the same fresh block, seeds 32000–32059. Raw:
`runs/env_007/rung_06m/{eval_06m_arms,eval_1m2,eval_06m_ladder}_block32000.json`,
`rung_analysis.json`, `readout_06m_all.json`.

### 5.1 The 0.6M rung does not rank candidates like 1.2M — it *inverts* the ranking

| candidate | 0.6M success, 3 seeds | seed sd | 0.6M dock | 1.2M success | 1.2M dock |
|---|---|---:|---:|---:|---:|
| control_v1 | 22.2 % (0.0–**65.0** %) | 0.371 | 0.528 | **0.0 %** | 0.567 |
| probeA | 10.6 % (1.7–20.0 %) | 0.092 | 0.494 | 43.3 % | 0.767 |
| probeB | 48.3 % (11.7–75.0 %) | 0.328 | 0.761 | 36.7 % | 0.917 |
| probeC | 13.9 % (0.0–41.7 %) | 0.241 | 0.694 | **98.3 %** | 0.983 |
| probeD | 7.8 % (0.0–23.3 %) | 0.135 | 0.461 | 73.3 % | 0.850 |
| probeE | 14.4 % (0.0–43.3 %) | 0.250 | 0.850 | 76.7 % | 0.867 |

Pre-registered statistics, all from `rung_analysis.py`:

* `rho(0.6M success, 1.2M success) = −0.486` (bar was `>= +0.60`);
* mean within-candidate seed sd **0.236** > between-candidate sd of means **0.149**;
* `rho(0.6M dock_rate, 1.2M dock_rate) = +0.543` — **the docking *behaviour* is partly
  preserved while the *success* is not**;
* `rho(0.6M mean_return, 1.2M mean_return) = −0.543`.

**Verdict: REJECT.** And the failure is in the dangerous direction: the 0.6M rung ranks
`control_v1` — the arm that is 0 % at 1.2M — **second best of six**, and puts the arm that
is 98.3 % at 1.2M (`probeC`) fourth. A search that selected on the 0.6M rung would have
preferentially kept the reward that reward-hacks.

The `dock_rate` line is the explanation. At 0.6M every arm enters the dock a lot
(0.46–0.85) and almost none satisfies the success predicate (heading < 30°, speed <
0.05 m/s, held 10 consecutive steps). So "the crate reached the dock" is *also* not a valid
cheap proxy — it is the near-miss that the strict predicate exists to exclude.

**"More seeds" cannot rescue this.** The numbers above are already 3-seed means, so adding
seeds shrinks the variance term (0.236) but not the *bias*: the estimates converge to the
same means, and the same means give rho ≈ −0.49. The rung is not noisy-but-centred; it is
systematically wrong.

### 5.2 The mechanistic readout, out of sample at 0.6M — also REJECTED

`mechanistic_readout.py` on all 26 runs (pre-registered thresholds, §3c of the
pre-registration):

```
confusion over the 14 candidates : tp=2  fp=0  fn=3  tn=9
precision 1.000   recall 0.400   AUPRC 0.877   (baseline: precision 0.357, recall 1.0, AUPRC 0.357)
VERDICT: REJECT  (bar: AUPRC >= 0.60 AND recall >= 0.60)
```

It is precise and incomplete: it fires on `probeA` and `probeB` and on nothing bad — all
nine negatives, including all eight LLM candidates and `control_v1`, are correctly
rejected. But `probeC`, `probeD` and `probeE` are missed, because at 0.6M their terminal
term has not yet taken over in a majority of seeds (probeC: seed 0 dominant
`terminal_success` 0.955, seeds 1–2 still stuck at `dock_enter` with share ≈ 0.42;
probeD 1/3 seeds; probeE 1/3 seeds). `control_v1` seed 0 *did* fire at the single-run
level (`dock_settled_hold` share 0.961, +166 per episode) and was saved only by the
majority vote — a real false-positive risk from the +20/step settled stream it contains,
which the readout cannot distinguish from `probeA`/`probeD`'s one-off settled event.

So at 0.6M the readout is a **safe screen but not a selector**: it never certified a bad
arm, and it also failed to certify 3 of the 5 known-good ones. (For contrast, the same
readout separates all 14 candidates at 1.2M — in-sample, and reported only as such.)

### 5.3 What the seed spread revealed: training *longer* can destroy the behaviour

The 3-seed spread was expected to be a nuisance. It turned out to be the finding. Comparing
seed 0 across budgets:

| arm | 0.6M, seed 0 | 1.2M, seed 0 | direction |
|---|---:|---:|---|
| control_v1 | **39/60** | **0/60** | **destroyed** |
| probeA | 12/60 | 26/60 | improved |
| probeB | 7/60 | 22/60 | improved |
| probeC | 25/60 | 59/60 | improved |
| probeD | 0/60 | 44/60 | improved |
| probeE | 0/60 | 46/60 | improved |

`control_v1` is the only arm that gets **worse** with more optimisation, and it is exactly
the arm whose reward is dominated by a farmable dense term (`crate_progress` 76 % +
`cart_approach` 24 % at 1.2M, with the `+20`/step settled stream no longer active). Its
0.6M policy *does* dock (39/60, `dock_settled_hold` share 0.96); further PPO training trades
that risky behaviour for the unconditionally available progress/approach reward and ends at
0/60. That is reward hacking emerging with optimisation — and it is the same phenomenon as
the pilot's 5/60 at 1.2M collapsing to 0/60 at 3M (`SESSION_STATE.md` §3c).

It also means a *short*-training selector is not merely low-resolution here: it would
systematically favour rewards whose apparent early success is destroyed by more training.

### 5.4 The LLM pool is dead at both rungs

All eight ladder candidates at 0.6M: **0/60, `dock_entered = 0.00`** — they do not even
reach the dock, and their component tables show nothing active, exactly as at 1.2M
(`eval_06m_ladder_block32000.json`). So there is no non-monotone surprise hiding in the
generated pool, and no selector — however good — could have rescued a pool with no positive
member.

### 5.5 Conclusion of the Branch B branch

No cheap instrument separates these rewards:

* structural checks on the function (training-free): no correlation, §1;
* trajectory ranking on scripted-reachable trajectories: perfect separation and still
  useless, §3f;
* realised component activity at **0.6M**: precise but recall 0.40;
* realised component activity at **1.2M**: the only instrument that separated anything
  (14/14) — at the same budget as simply training the candidate and measuring it.

**The cheapest rung at which any selector works is the budget of the thing being selected,
so selection cannot be made cheaper than evaluation on this environment.** The "cheap
selector" programme is closed from the selection side, which independently agrees with the
generation-side reading this session reached (`SESSION_STATE.md` §5): the bottleneck is that
the generator never emits a reward whose global landscape PPO can follow.

§5.6 and §5.7 then test the one rung this argument leaves open — 1.0M — with both
instruments, and it fails too; §5.7 is the consolidated verdict table.

### 5.6 The intermediate rung (1.0M): pre-registered, and also rejected

`LADDER_DECISION_PREREGISTRATION.md` §3d fixed the prediction and the bar before these runs
existed: if the 0.6M failures are mid-transit instability, the readout should recover at
1.0M, and would be adopted iff it certified ≥ 3 of the 5 positive arms **and** did not
certify `control_v1`. 18 runs (six arms × seeds {0,1,2}) at 1.0M, `runs/env_007/rung_10m/`.

**Outcome-based rung** (`rung_analysis.py --short-label 1.0M`):

| candidate | 1.0M success (3 seeds) | seed sd | 1.2M success |
|---|---|---:|---:|
| control_v1 | 32.8 % (10.0–73.3 %) | 0.352 | **0.0 %** |
| probeA | 14.4 % (5.0–20.0 %) | 0.082 | 43.3 % |
| probeB | 76.1 % (63.3–100 %) | 0.207 | 36.7 % |
| probeC | 82.2 % (73.3–96.7 %) | 0.126 | **98.3 %** |
| probeD | 17.8 % (0.0–51.7 %) | 0.294 | 73.3 % |
| probeE | 43.9 % (0.0–90.0 %) | 0.450 | 76.7 % |

`rho(1.0M success, 1.2M success) = +0.371` (bar ≥ +0.60) → **REJECT**. Note what changed:
the correlation is now *positive* (it was −0.486 at 0.6M) and the noise condition is nearly
met (within-candidate sd 0.252 < between-candidate sd 0.289), so the rung is improving with
budget — it simply has not arrived. `control_v1`, 0 % at 1.2M, is still ranked **4th of 6**.

**Mechanistic readout at 1.0M** (`--require-negative control_v1`): precision 0.833,
**recall 1.000**, AUPRC 0.877 (baseline 0.833) → **REJECT** on the §3d condition, because
`control_v1` is certified positive in **3/3** runs.

This is informative in two directions:

* the 0.6M diagnosis was right about *sensitivity*: recall goes 0.40 → 1.00 once the terminal
  term has stabilised, confirming that the 0.6M misses were mid-transit, not a broken
  instrument;
* but the failure simply moves. At 1.0M `control_v1`'s settled stream is active and dominant
  in every seed (share 0.971, +238 per episode), and the readout cannot distinguish a
  farmable settled **stream** from a one-off settled **event**. At 1.2M the same arm is
  rejected — not because the instrument got better, but because its policy has by then
  abandoned docking, which switches its settled term off. **The 14/14 at 1.2M is therefore
  partly an accident of that arm's degeneration at that budget.**

Per-seed, the collapse is monotone for the arm that collapses: `control_v1` seed 0 runs
39/60 at 0.6M → 9/60 at 1.0M → 0/60 at 1.2M, while the hand-designed arms do not show that
pattern. The "training longer destroys the behaviour" reading therefore holds at the
intermediate rung too, not only between 0.6M and 1.2M.

### 5.7 Verdict on the rung question

| rung | instrument | result |
|---|---|---|
| 0.6M | outcome (success) | `rho = −0.486`; seed noise > between-candidate spread → REJECT |
| 0.6M | component readout | precision 1.00, **recall 0.40** → REJECT |
| 1.0M | outcome (success) | `rho = +0.371`; noise ≈ spread → REJECT |
| 1.0M | component readout | recall 1.00 but **certifies `control_v1`** → REJECT |
| 1.2M | component readout | 14/14 — in-sample, and partly luck (see §5.6) |

Both instruments improve monotonically with budget and neither is usable before the
reference budget. So the conclusion stands, now on three rungs and two instruments:
**the cheapest rung at which any selector works is the budget of the thing being selected.**

One honest caveat on the reference itself: the labels are 1.2M outcomes, and §3c shows that
1.2M → 3M can *change* an outcome (the pilot's 5/60 → 0/60). "1.2M is the truth" is a
convention of this project, not an established fact — which makes the selector problem
harder still, because if even the target moves with budget, no fixed rung can be validated
against it.
