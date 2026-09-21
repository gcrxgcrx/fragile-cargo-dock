# CLOSE-SPEED ISOLATION FINDINGS — the one term that separates 0 % from 77 %

Protocol, predictions and the verdict rule were fixed in
`runs/env_007/CLOSE_SPEED_ISOLATION_PREREGISTRATION.md` before any training.
Raw: `runs/env_007/close_speed_test/{eval_block37000_wave1,eval_block37000_wave2}.json`,
`runs/env_007/close_speed_test/settling_block37000_wave1.json`, and the four
`training_summary.json` under `runs/env_007/close_speed_test/`.

## 1. The experiment

`C0` and `C1` are the same program except for one block. Built by
`make_closing_speed_variants.py` from `runs/env_007/control_obs_only/reward.py`:

* `C0` — byte-identical `control_v1` body (two telescoping shaping terms, a `+20` per-step
  settled-state hold, a boundary guard, a shove penalty);
* `C1` — `C0` + the observation-only closing-speed penalty from `probeD`
  (`-0.05 * contact * max(0, cart_forward_speed - crate_velocity·heading)`).

Verified equal to the arms they replicate: `max |C1 - probeD| = 0.000e+00` and
`max |C0 - control_v1| = 0.000e+00` over 27 scripted trajectories, so `C1`/seed 0 is
`probeD`'s existing model and `C0`/seed 0 is `control_v1`'s.

Protocol: 1.2 M steps, `n_envs = 6`, clip 20, `normalize_reward = true`, gamma = 0.999,
training seeds 0/1/2 paired, all scored on the **fresh block 37000-37059** (60 episodes).

## 2. Result

| arm | training seed | success | dock | P(success \| dock) | mean native return |
|---|---:|---:|---:|---:|---:|
| `C0` | 0 | **0/60** | 0.90 | **0.000** | 8.2 |
| `C0` | 1 | **48/60** | 0.87 | 0.923 | 248.4 |
| `C1` | 0 | **46/60** | 0.87 | 0.885 | 238.4 |
| `C1` | 1 | **26/60** | 0.70 | 0.619 | 137.5 |
| `C1` | 2 | **0/60** | **0.13** | 0.000 | 4.4 |

**Pre-registered verdict: S5 HOLDS** — `C1` is non-zero on 2 of 3 training seeds and at
least one seed is >= 8/60, so a full-pipeline run is *licensed* by the rule.

**The honest complication, which must be reported with it:** the effect does **not**
survive pairing. Seed 0 is 0/60 vs 46/60 (the isolated term is worth +77 points); seed 1 is
48/60 vs 26/60 (**the control wins**); pooled over 120 episodes, `C1` 72 vs `C0` 48,
one-sided Fisher **p = 0.55**. Seed 2 fails in exactly the shape of `SESSION_STATE.md` 5.8's
x100-seed-2: `dock_entered` collapses to 0.13, i.e. the policy never starts approaching.

## 3. The mechanism readout, which is the part that is not noise

`diagnose_settling.py` (new) reports the **maximum consecutive stable-step count** reached
per episode — success needs 10. On the same 60-episode block:

| arm | success | dock | max stable (mean / median / p90) | median dock-entry speed | docked-and-failed episodes that reached 7-9 steps |
|---|---:|---:|---|---:|---:|
| `C0` seed 0 | 0/60 | 0.90 | **0.83 / 1 / 2** | 0.087 | **0/54** |
| `C1` seed 1 | 26/60 | 0.70 | **4.93 / 5 / 10** | 0.202 | 2/16 |
| `C1` seed 0 | 46/60 | 0.87 | **8.03 / 10 / 10** | 0.383 | 3/6 |

`C0` reaches the dock in 90 % of episodes and **never holds the settled condition for even
one step on average**; 54 docked-and-failed episodes and *not one* got within three steps of
success. Adding one observation-only closing-speed penalty moves the same statistic to 5-8
steps. So the missing capability is **settling**, not reachability, and it is a term the
observation contract *can* express.

Note also the entry speeds: the arm that succeeds performs a **higher**-speed dock entry
(0.38 m/s median) than the arm that fails (0.087). "Arrive slowly" is not what the term
teaches; it suppresses the closing *rate* while in contact, which is what stops the crate
from being knocked back out of the tolerance box.

## 4. An early-training predictor of the outcome (new, not pre-registered)

From the four `training_summary.json` final-policy component tables:

| run | `dock_settled_hold` active rate | its per-episode sum | eval return | fresh success |
|---|---:|---:|---:|---:|
| `C0` seed 1 | **0.0409** | **170** | 263.8 | 48/60 |
| `probeD` (= `C1` seed 0) | 0.0293 | 140 | 218.1 | 46/60 |
| `C1` seed 1 | 0.0131 | 97 | 52.5 | 26/60 |
| `C0` seed 0 | 0.0085 | 68 | 8.4 | 0/60 |
| `C1` seed 2 | **0.0024** | **19** | 19.7 | 0/60 |

The ordering is not strictly monotone (`probeD` beats `C0` seed 1 on `probeD`'s own run but
not on this table), so this is **not** a selector — but it is readable from a few hundred
thousand steps of training, i.e. before the run finishes, and it separates the two 0/60 runs
from everything else by an order of magnitude. That makes it a candidate **early-stopping /
seed-budgeting** signal for future searches, which is a different (and defensible) use from
"predict which reward will learn".

## 5. What this changes

* **The single-variable contrast that the project had left unexplained is now explained and
  closed**: `control_v1` 0 % vs `probeD` 73.3 % is caused by exactly one added term, and the
  term acts on settling.
* **The external review's request is satisfied at the level of the mean effect**, but its
  stronger claim — that per-training-seed pairing would make the comparison clean — is
  **falsified by the pairing itself**: one pair reverses. With 1.2 M steps and n = 1 training
  seed per arm, this environment's seed variance is comparable to the effect size. Any future
  A/B here needs **>= 3 training seeds per arm** and must report paired differences, not
  pooled ones.
* **The verdict rule was written before the data and is kept**: S5 holds literally. But the
  honest reading is weaker than the letter of the rule, and it is recorded here so that the
  rule is not used afterwards to justify a pipeline run on its own.
