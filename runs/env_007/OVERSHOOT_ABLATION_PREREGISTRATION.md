# Pre-registration: is "make overshoot very negative" the right way to teach the release lead time?

Written before any arm was trained. Motivation: the user's pushing-task experience — *the AI
never learns the release lead time, so make the score for pushing past the dock extremely low
and let it find the near-but-not-past point*. The repo's own
`record_overshoot_diagnosis.py` supports the diagnosis: the CREATE policy looks "inches away"
under the 400-step limit, and **blows straight through the dock** once the limit is lifted,
while the native-reward policy releases early and settles.

## Base and geometry

Base reward: `runs/env_007/repair_test/r09_progress_x50_stream/reward_v1.py` — the measured
two-edit repair of `L2/cand_00`, **23/60 = 38.3 %** with `dock_entered` 0.98. Patching a
*working* reward makes this a clean one-axis ablation with a matched baseline.

Verified in `custom_envs/fragile_cargo_dock_env.py`: cart spawns at x ∈ (−3.5, −2.8), crate at
x ∈ (−1.9, −1.1), dock at `DOCK_X = 2.6` → the push direction is **+x**, so
`obs[12] = (cargo.x − 2.6)/5.0 > 0` is **past the dock centre**. x tolerance is 0.024.

## Arms (5, one 1.2M run each, seed 0; scored on fresh seeds 35000–35059)

| arm | edit relative to `r09` |
|---|---|
| `o00_base` | byte-identical copy — matched baseline on the same block |
| `o01_overshoot` | + hinge penalty on overshoot, weight **−20** (the per-step clip ceiling) |
| `o02_gentle` | + probe D's **closing-speed** penalty (−0.05·contact·closing) |
| `o03_both` | + overshoot hinge **and** gentleness |
| `o04_overshoot_x10` | + the same hinge at **−200**, i.e. 10× past the clip |

The hinge: `ov = clip(next_obs[12] − 0.024, 0, 0.04)`, term `= w · ov/0.04` (0 inside the
tolerance, saturating at `w` once 0.04 m past). Because the harness clips the **per-step
total** to ±20, `w = −200` is delivered as −20, so `o04 ≡ o01` by construction — that
equivalence is itself part of what this experiment tests.

## Training-free predictions, computed before training

`advantage_scale_probe.py` on the five reward files:

| arm | `A` (per step) |
|---|---:|
| `o00_base` | **+0.141** |
| `o02_gentle` | **+0.138** |
| `o01_overshoot` | **−0.253** |
| `o03_both` | −0.256 |
| `o04_overshoot_x10` | −0.304 |

The overshoot penalty **flips the working reward's advantage negative** (acting becomes worse
than idling) because the scripted controllers overshoot on many steps, so the hinge fires
along the very trajectories that reach the dock. Gentleness, by contrast, leaves `A` intact.

## Predictions

| # | prediction |
|---|---|
| **U1** | if the user's mechanism is right, `o01 > o00` (the overshoot cliff teaches the lead time and improves on 23/60) |
| **U2** | my measured-evidence reading: `o01 ≤ o00`, and specifically `o01 ≈ 0/60` with a **falling** `dock_entered`, because `A < 0` makes inaction optimal — the same signature as `r11` (own speed penalty ×100: 0/60, dock 0.00, `A` = −12.8) |
| **U3** | `o02 ≥ o00`: the closing-speed penalty is the measured release-lead-time signal (probe D = control v1 + gentleness, 73.3 %, vs control v1 0 %) and it does not damage `A` |
| **U4** | `o04 ≈ o01` (within seed noise): the clip makes a 10× harsher penalty equivalent, so "拉到极低" cannot buy rank beyond one clip step |
| **U5** | mechanism: in any arm where `dock_entered` falls below `o00`'s, the failure is *avoidance* (the policy declines to approach), not fine control |

## Interpretation, fixed now

* **U1 holds** → the user's guidance is the right one, the two-edit recipe is incomplete, and
  the next prompt revision should mandate an overshoot penalty.
* **U2 holds** → the diagnosis (lead time is the crux) is right but the *implementation* is
  what matters: a positional cliff on the far side punishes the only physically available route
  to the dock, so it teaches avoidance; the lead-time skill must be taught by a **dense penalty
  on closing speed**, which is what the working hand-written arm uses. The prompt should
  therefore ask for the *rate* signal, not a positional cliff — and the ±20 clip means an
  extreme overshoot penalty is not even expressible.
* Either way this is a **single-axis, matched-block** comparison on a reward that is known to
  deliver, so it decides the question without depending on any cross-prompt comparison.

Standing caveat: oracle-authored ablation; it measures which guidance signal works, not whether
a search would find it.

---

## OUTCOME (fresh seeds 35000–35059, 60 episodes; one matched block, no cross-block comparison)

| arm | `A` (predicted before training) | fresh-60 | dock | mean return |
|---|---:|---:|---:|---:|
| `o00_base` (copy of the 23/60 reward) | +0.141 | **16/60 = 26.7 %** | **0.90** | 88.65 |
| `o01_overshoot` (−20/step cliff) | −0.253 | **0/60** | **0.00** | −1.47 |
| `o02_gentle` (+ closing-speed penalty) | +0.138 | **1/60** | 0.35 | 10.70 |
| `o03_both` | −0.256 | **0/60** | **0.00** | −0.80 |
| `o04_overshoot_x10` (−200/step cliff) | −0.304 | **0/60** | **0.00** | −7.95 |

| # | prediction | outcome |
|---|---|---|
| **U1** | the user's mechanism: `o01 > o00` | **FAILS** — 0/60 vs 16/60, and `dock_entered` collapses 0.90 → 0.00 |
| **U2** | `o01 ≤ o00`, specifically `o01 ≈ 0/60` with a **falling** dock rate (avoidance) | **HOLDS** — and the training-free probe predicted it before the run (`A`: +0.141 → −0.253) |
| **U3** | `o02 ≥ o00` (gentleness is the measured release-lead-time signal) | **FAILS** — 1/60 vs 16/60, dock 0.90 → 0.35 |
| **U4** | `o04 ≈ o01`: the ±20 per-step clip makes a 10× harsher cliff equivalent | **HOLDS** — both 0/60, dock 0.00; the harsher one is no better (return −7.95 vs −1.47) |
| **U5** | failures show *avoidance* (`dock_entered` falls), not fine control | **HOLDS** in all three failing arms (dock 0.00) |

(The `o00_base` copy scores 16/60 here against the same file's 23/60 on block 32000–32059. That
~7-point gap is block noise — the reason every comparison above is inside one block.)

### Why the overshoot cliff destroys a working reward

Measured mechanism, not interpretation: the cart has no brake, so the **only** route to the
dock is to push the crate and let drag stop it — trajectories that arrive therefore often pass
through or near the far edge. Penalising "past 0.024" at −20/step makes the approach itself
unprofitable, `A` goes negative, and the optimum becomes *not approaching*: `dock_entered`
0.90 → 0.00 from a single added term. Same signature as `r11` (own speed penalty ×100: 0/60,
dock 0.00, `A` = −12.8) and as the whole v5/v7 immobile family (`A ≤ 0`).

### The gentleness result, now replicated three times

Adding `probeD`'s closing-speed penalty **hurts** a reward that already penalises closing speed
(`r09` contains `soft_contact = −1.2·contact·closing`): `r08` (×50 + gentleness) 0/60 dock 0.00;
`r10` (×50 + stream + gentleness) 0/60 dock 0.27; `o02` here 1/60 dock 0.35 against a 16/60
baseline in the same block. So the correct statement is not "gentleness is the decisive
ingredient" but: **it is decisive only relative to a reward that lacks it (probe D vs
control v1); stacked as an extra term on a reward that already has it, it suppresses the push
and destroys delivery.**

### Consequence for the prompt

Neither "guidance" signal a domain expert would reach for first — a positional overshoot cliff,
or an extra gentleness penalty — survives contact with a *working* reward on this task. Both are
now measured against a matched baseline, and the advantage probe flagged the overshoot one
before any training happened. The remaining candidate forms are those under test in
`APPROACH_ABLATION_PREREGISTRATION.md` (proximity vs delta vs the near-and-slow funnel).
