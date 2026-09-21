# SEED-REPLICATION FINDINGS — the success rate is seed-dominated, and longer training destroys it

Pre-registered in `runs/env_007/SEED_REPLICATION_PREREGISTRATION.md` (decision rule and
analysis fixed before the runs). Raw: `runs/env_007/close_speed_test/eval_block38000_seeds.json`,
`.../eval_3m_C0seed0_block38000.json`, `analyze_seed_replication.py`.

## 1. The result

`C0` = `control_v1` (hand-written, observation-only: two telescoping shaping terms, a `+20`
per-step settled hold, a boundary guard, a shove penalty).
`C1` = `C0` + one closing-speed penalty. 1.2 M steps, training seeds 0-3, **all scored on the
fresh block 38000-38059**.

| arm | seed 0 | seed 1 | seed 2 | seed 3 | mean | sd | dock range |
|---|---:|---:|---:|---:|---:|---:|---|
| `C0` | 0 | 43 | **54** | 35 | **33.0 / 60** | 23.3 | 0.58-0.93 |
| `C1` | **45** | 21 | 0 | 13 | **19.8 / 60** | 18.9 | 0.02-0.80 |

Paired per-seed differences: **+45, −22, −54, −22** -> mean **−13.2**, sd 41.7,
**95 % paired t-interval [−79.5, +53.0]**. Pre-registered verdict: **N (null)** — the interval
contains 0, so the isolation effect is **not established at the seed level**, and the +77-point
single-seed contrast reported in `CLOSE_SPEED_ISOLATION_FINDINGS.md` was a draw from a very
wide distribution.

## 2. The finding that matters more than the null

**The seed spread is not a property of LLM-written rewards.** The hand-written control —
the arm used as the project's "reference for what works" — scores **0, 43, 54 and 35** out of
60 across four training seeds at 1.2 M steps, with `dock_entered` between 0.58 and 0.93.
`C1` spans 0 to 45. So:

* a single training seed can be off by ~50 successes out of 60 relative to another seed of the
  *same* reward and *same* hyperparameters;
* every single-seed comparison in this project — including its headline "the two-edit repair
  reaches 38.3 %" and every ablation in §5.6-5.8 — is one draw from that distribution;
* the question "why do LLM rewards score 0-5 % while hand-written rewards score 40-77 %" is
  partly **mis-posed**: at 1.2 M the *same* hand-written reward scores 0 % on one seed and
  90 % on another. The LLM/hand-written contrast was measured on samples from overlapping
  distributions.

## 2b. The budget axis, measured both ways (added 06:22)

`C0` was trained at two budgets for two seeds and scored on the **same** block 38000-38059:

| arm / seed | 1.2 M | 3.0 M |
|---|---:|---:|
| `C0` seed 0 (the "bad" seed) | **0/60** | **2/60** (dock 0.90) |
| `C0` seed 1 (the "good" seed) | **48/60** | **51/60** (dock 0.87) |

Together with the existing record for the same arm at lower budgets
(`LADDER_FINDINGS.md` §5: 0.6 M `39/60`, 1.0 M `22.2 %` with a seed range of 0-65 %), the
picture is:

* **the seed decides, and it decides early.** A seed that fails at 1.2 M still fails at 3 M
  (0 -> 2/60); a seed that works at 1.2 M still works at 3 M (48 -> 51/60). It is not
  "slower learning" — the bad seed's component table at 3 M shows *more* settled-state
  activity than at 1.2 M (`episode_sum_mean` 112 vs 68) while delivery stays at zero.
* therefore the per-seed spread is **not** narrow at 3 M either: within one arm the two seeds
  are 2/60 and 51/60 at the same budget.
* and the arm itself has a ceiling: the 1.0 M record for `C0` (seed range 0-65 %, mean 22 %)
  versus the good seed's 51/60 at 3 M says the arm's *best* seed is far above its mean.

**What this buys the pipeline.** Because a seed's class is already visible at 1.2 M, a
**single cheap run can eliminate a hopeless candidate** (a 0/60 at 1.2 M did not become a
success in either seed tested). What a single run *cannot* do is **select a winner** — the
project's selection rule keeps the candidate with the best single-seed score, which
systematically keeps the lucky draws of the arm class that cannot succeed and discards the
unlucky draws of the class that can.



From the existing record (`runs/env_007/LADDER_FINDINGS.md` §5, `LADDER_FINDINGS.md:230`),
for `control_v1` seed 0:

| budget | fresh success |
|---|---:|
| 0.6 M | **39/60** |
| 1.0 M | 22.2 % (0-65 % over seeds) |
| 1.2 M | **0/60** |
| 3.0 M | **2/60** (this session, block 38000-38059; dock 0.90) |

**The arm gets worse with more optimisation.** It learns to deliver at 0.6 M and then
optimises its way out of it (the documented mechanism is reward hacking emerging with
training). Its 3 M component table shows this directly: the settled-hold term is *more* active
than at 1.2 M (`episode_sum_mean` 112 vs 68) while delivery goes to zero — the policy is
farming the settled state rather than completing.

And the "bad seed" does **not** recover: `C0` seed 0 at 3 M is still 2/60 at dock 0.90. The
basin the seed lands in is persistent in the direction that matters.

## 2c. The addendum experiment: the seed x budget interaction (added 06:53)

The prediction recorded before these runs was: a 1.2 M score predicts a seed's class at 3 M.
**It is falsified for one arm and confirmed for the other.**

| arm / seed | 1.2 M (block 38000-38059) | 3.0 M (same block) |
|---|---:|---:|
| `C0` seed 0 | 0/60 | **2/60** |
| `C0` seed 1 | 48/60 | **51/60** |
| `C1` seed 0 | **45/60** | **3/60** (dock 0.88) |
| `C1` seed 2 | 0/60 (dock 0.02) | **0/60** (dock 0.92) |

* `C1` seed 0 **collapses** from 45/60 to 3/60 when trained three times as long — while its
  docking *improves* (dock 0.02 -> 0.92 for seed 2, and 0.80 -> 0.88 for seed 0). Both arms
  reach the dock at 3 M; almost none of them settle.
* Across the four (arm, seed) cells measured at 3 M, three score **0-3/60** and one scores
  **51/60**. The 3 M regime is therefore **not** "the good budget": it is the regime in which
  the delivery behaviour is optimisation-fragile, and one lucky seed out of four survives it.
* So the honest statement is **not** "a 1.2 M run predicts the seed's class" (falsified) and
  not "3 M is better" (it is worse for 3 of 4 cells). It is: **success is a joint property of
  (reward, training seed, budget), and at both budgets most cells fail.**

This also means the project's own paper tables, which report one number per method at 3 M
(`0/20` each), and the hand-written arms' 3 M numbers, are each **one cell** of this table.


## 4. Consequences

1. **The search cannot be judged on one seed, and the pipeline's own selection is affected.**
   CREATE/EUREKA select by `mean_eval_reward` on 20 training-seed episodes with one training
   seed per candidate. A candidate's measured score is dominated by its optimisation draw.
2. **The published "0/60 vs 23/60" repair result needs a seed-level re-statement.** The
   recipe is real (two edits, mechanism confirmed), but its 38.3 % was measured once.
3. **`>= 3 training seeds per arm** is now the minimum for any claim in this project, and the
   report must give per-seed values, not only means.
4. **The open problem is no longer "what ingredient is missing from the reward"** — for the
   failing family the answer was settling, and even supplying it does not beat seed noise. The
   open problem is **how to make a reward whose good behaviour survives optimisation**, on an
   environment where the same hand-written reward is destroyed by 2x more training.
5. Practical implication for the full pipeline: running CREATE vs EUREKA at 10 x 3 M per
   method measures a regime in which the *hand-written* reference arm scores 2/60. If a method
   comparison is run there, both methods will most likely be at 0/20 and the comparison will
   be degenerate — which is what the earlier run found. **The fidelity decision (1.2 M vs
   3 M) may matter more than the method.**

## 5. The same variance is present in every arm ever trained (existing data, re-analysed)

The `rung_06m` / `rung_10m` evaluations kept per-seed records for six arms, three training seeds
each, on one block (32000, 60 episodes). Re-read here:

| arm | 0.6 M per-seed | mean | 1.0 M per-seed | mean |
|---|---|---:|---|---:|
| `control_v1` | 39 / 1 / 0 | 13.3 | 9 / 6 / 44 | 19.7 |
| `probeA` | 12 / 6 / 1 | 6.3 | 3 / 11 / 12 | 8.7 |
| `probeB` | 7 / 45 / 35 | 29.0 | 38 / 60 / 39 | 45.7 |
| `probeC` | 25 / 0 / 0 | 8.3 | 44 / 58 / 46 | 49.3 |
| `probeD` | 0 / 14 / 0 | 4.7 | 0 / 31 / 1 | 10.7 |
| `probeE` | 0 / 0 / 26 | 8.7 | 54 / 0 / 25 | 26.3 |

* **The spread is an order of magnitude larger than the arm differences.** `probeD` — the
  project's 73.3 % arm — reads **0 / 14 / 0** at 0.6 M and **0 / 31 / 1** at 1.0 M.
  `probeC` — the 98.3 % arm — reads 25 / 0 / 0 at 0.6 M. `probeB` reads 7 / 45 / 35.
* So "which hand-written reward works" and "which LLM reward works" are **overlapping
  distributions at these budgets**, and the project's headline contrasts (`probeD` 44 vs an LLM
  candidate 0) are single draws from them.
* With n = 3 seeds, the standard error of an arm's mean is roughly ±7-20 successes out of 60 —
  i.e. larger than most of the differences the project has been reasoning about, including
  several of the nine guidance ablations in §5.6.

**This does not invalidate the mechanism findings** (they are contrasts on a *statistic* — max
consecutive stable steps, component activation — measured on the same trained policy, and they
separate cleanly), but it does mean that **every success-rate claim in this project needs a
per-seed restatement**, and that the selection problem is dominated by an optimisation effect
rather than by the reward text.

## 6. Does this force the pipeline to use more seeds? A rough cost reading


Not necessarily, and the arithmetic is worth stating because it decides the pipeline's design.

Total variance of a candidate's measured success = between-training-seed variance + binomial
evaluation noise. Taking the measured per-arm spreads at 1.2 M (`C0` sd 23.3, `C1` sd 18.9 out
of 60) and the binomial sd at 60 episodes near p ≈ 0.3 (≈ 4/60), the seed component dominates
by roughly a factor of 5 in sd, i.e. **~25x in variance**. To average it down to the level of
the evaluation noise at 1.2 M would take on the order of 16-25 training seeds per candidate.

Two ways out, both cheaper than that:

* **Evaluate a candidate at several training seeds and treat the seed as the unit** (what this
  experiment did: 4 seeds is enough to detect a 45-point effect, not a 15-point one);
* **Use the highest-fidelity budget available** if the collapse is budget-driven — but §3 shows
  3 M is *worse* for the reference arm, so fidelity is not a free fix here.

The cheapest defensible pipeline design is therefore: **screen candidates at 1.2 M with 1 seed
to drop the structurally broken ones, then confirm the survivors with >= 3 training seeds**, and
report the per-seed values rather than a mean. That is a different design from the one the
current CREATE/EUREKA configs use.

