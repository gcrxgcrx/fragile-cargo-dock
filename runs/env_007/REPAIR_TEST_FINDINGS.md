# REPAIR TEST FINDINGS — the first LLM-derived reward that ever delivered the crate

Companion to `runs/env_007/REPAIR_TEST_PREREGISTRATION.md` (protocols and predictions fixed
before each wave), `runs/env_007/ALIGN_FIX_PREREGISTRATION.md`,
`runs/env_007/ADVANTAGE_PROBE_FINDINGS.md` and `runs/env_007/LADDER_FINDINGS.md`.

Base candidate: `runs/env_007/prompt_ladder/L2/cand_00/reward_v1.py` — structurally the
strongest LLM reward produced in this project (one-off terminal event ✔, terminal dominance
✔, gentleness gap 1.14) and **0/60** on fresh seeds, like all ~35 LLM rewards before it.

Protocol for every arm: `configs/env007_terminal_rule_pilot.yaml`, `n_envs=6`, clip 20,
**1.2M** steps, seed 0; scored with `eval_pool.py` on fresh seeds **32000–32059** (60
episodes), with matched re-scores of the base models so every comparison is inside one seed
block. Raw JSONs and per-arm component tables are in
`runs/env_007/repair_test/`, `runs/env_007/align_fix_test/`, `runs/env_007/recipe_replication/`.

**This is an oracle-authored repair study.** It measures whether a repairable target exists
and how large the fix is; it is *not* evidence that a search would find the edit
(`SESSION_STATE.md` §4 error #7).

---

## 1. THE RESULT

| arm | edit relative to `L2/cand_00` | fresh-60 | dock rate | mean native return |
|---|---|---:|---:|---:|
| base `L2/cand_00` | — | 0/60 | 0.00 | −1.20 |
| `r00_copy` | none (byte-identical) | 0/60 | 0.00 | −1.20 |
| `r01_guidance` | + probe D's dense guidance | 0/60 | 0.00 | −0.40 |
| `r02_guidance_gentle` | `r01` + gentleness | 0/60 | 0.00 | −0.08 |
| `r03_guidance_x3` | `r01`, guidance ×3 | 0/60 | 0.00 | +0.39 |
| `r04_own_progress_x50` | **its own approach coefficient 12 → 600** | 0/60 | **0.97** | +9.14 |
| `r05_guidance_no_events` | `r02`, own events removed | 0/60 | 0.00 | −0.08 |
| `r06_settled_stream` | + dense `+20·done_cond` payoff stream | 0/60 | 0.00 | −1.20 |
| `r07_stream_and_guidance` | `r02` + the stream | 0/60 | 0.00 | −0.08 |
| `r08_progress_x50_gentle` | `r04` + gentleness | 0/60 | 0.00 | +1.67 |
| **`r09_progress_x50_stream`** | **`r04` + the dense payoff stream** | **23/60 = 38.3 %** | **0.98** | **+124.27** |
| `r10_x50_stream_gentle` | `r04` + stream + gentleness | 0/60 | 0.27 | +5.03 |
| `r11_x50_own_speedpen_x100` | `r04` + its own speed penalty ×100 | 0/60 | 0.00 | −1.65 |

`r09` is the **first LLM-derived reward in this project to deliver the crate**: 23/60 on
held-out seeds, one-sided Fisher **p = 8.69e-09** against the 0/60 that every other LLM
reward scores, and above the previous best LLM result ever recorded (5/60 = 8.3 %, itself not
significant: two-sided p = 0.0573). Its 38.3 % sits between hand-written `probeB` (36.7 %)
and `probeA` (43.3 %).

### The recipe, in two parts

`r09` differs from the base candidate in exactly two ways:

1. **its own approach coefficient ×50** (`crate_to_dock_progress`, 12 → 600) — this makes
   *acting* profitable; measured effect: `dock_entered` 0.00 → 0.97, mean final goal
   distance 4.148 m → 0.140 m;
2. **a dense payoff on the instantaneous success predicate** (`+20·done_cond` per step,
   where `done_cond` = inside ∧ aligned ∧ slow) — this makes *settling* profitable.

Neither half works alone: `r04` (scale only) docks but never settles → 0/60;
`r06` (stream only) never fires the stream because the predicate is never reached → the run
is **bit-for-bit equivalent to the copy control** (identical reward, identical policy,
identical 0/60).

### Mechanism, confirmed (prediction S5)

| quantity | base / `r06` | **`r09`** |
|---|---:|---:|
| `success_event` active rate | **0.0000** | **0.0012** |
| episode terminations (in-training, 20 eps) | 0 | 9 |
| realised mass shares | `dock_speed_penalty` 94 % | `settled_stream` 58 %, `progress` 20 %, **`success_event` 12 %**, `dock_speed_penalty` 8 % |

The candidate's own one-off +300 payoff was always correctly written — it simply had never
been reached. Once the policy can reach the predicate it fires and contributes 12 % of the
mass.

### Controls that make the result interpretable

* **Gentleness hurts here.** `r08` (×50 + gentleness) = 0/60 with `dock_entered` **0.00** —
  adding probe D's gentleness *destroyed the locomotion*; `r10` (×50 + stream + gentleness)
  = 0/60, dock 0.27. So the term that is decisive in the hand-written family (probe D) is
  actively harmful when stacked on this candidate's own `soft_contact` and
  `dock_speed_penalty`.
* **Raising penalties is not the fix.** `r11` (own speed penalty ×100) = 0/60, dock 0.00,
  return −1.65 — worse than the base. This was **predicted before its result was read** by
  the advantage-scale probe (`A` = −12.8 vs −0.26 for the base): scaling a penalty makes
  acting ruinously unprofitable.
* **The untouched copy reproduces the base exactly** (`r00`: 0/60, return −1.20, goal
  distance 4.148), so the generator's edit path is validated and every difference above is
  attributable to the recorded edit.

### Why it fails without the fix: a sign problem, not a structural one

`trajectory_ranking_check.py --describe` on the base reward: the scripted controllers earn
`idle −0.80`, `push_forever +6.35`, `release_0.25 +56.40` per episode. The **ordering is
already correct** — success ≫ pushing ≫ idling — and the trained policy nevertheless sits at
≈ −0.07 per episode and never moves the crate. Per-step, the advantage of acting is only
`(6.35 − (−0.80))/400 ≈ 0.018`, and the advantage-scale probe puts the whole reward at
**A = −0.259 per step**: *acting is worse than doing nothing*, so the policy's apparent
failure to learn is in fact correct optimisation of a mis-scaled reward. Multiply the
approach term by 50 and `A` becomes **+0.113** — and the cart moves.

**So the failure decomposes into two independent layers, and the fix into two independent
edits**: (i) reaching the dock region — a *scale* problem, one coefficient; (ii) settling
inside it — a *payoff-structure* problem, one term whose existing one-off form must become
dense.

## 2. Wave 2: the parts that did nothing, and a clean §3f replication

Beyond the arms tabled above:

* `r06`/`r07` — converting the one-off payoff into a stream **without** first making acting
  profitable does nothing at all, because the stream is gated on the same unreached
  predicate (`r06` ≡ base exactly).
* Adding probe D's guidance backbone — the thing a *working* hand-written arm has — leaves
  the crate exactly where it started (goal distance 4.148 m unchanged to three decimals) and
  `success_event` activation at 0.0000, even at 3× weight. Copying a working arm's terms
  into a failing candidate does **not** transfer its behaviour.

### The align-fix experiment: root cause found, fix does not rescue (`L0/cand_02`)

`L0/cand_02`'s alignment term is **inverted** (`heading_align = 1 − |obs[10]|`, while
`DOCK_ANGLE = 0` and success needs `|cargo_angle| < 30°`): it rewards the crate being
perpendicular to the dock, so its 94 %-of-mass `crate_docking_quality` term is maximised in
a state that can never succeed — textbook reward hacking with a one-token root cause,
verified against the environment source in `ALIGN_FIX_PREREGISTRATION.md`.

| arm | edit | fresh-60 | dock | return |
|---|---|---:|---:|---:|
| `f00_copy` | none | 0/60 | 0.00 | +1.32 (= original) |
| `f01_align_fix` | **one subscript**: `obs[10]` → `obs[11]` | **0/60** | 0.00 | −1.53 |
| `f02_align_fix_units` | `f01` + crate speed restored to m/s | 0/60 | 0.00 | −3.43 |
| `f03_align_fix_gentle` | `f01` + gentleness | 0/60 | 0.00 | −1.04 |

The training-free ordering check improves from 0.038 to **0.702** for that one subscript —
and the arm still never enters the dock. This is a **second, independent instance of §3f**
("ranking correctness ≠ learnability"), and a stronger one than the original: here the root
cause was identified, verified against the environment and repaired, and the reward still
does not train, because its per-step advantage stays at `A = −0.0004` (acting ≈ idling).

## 3. The recipe replicates on a second, independently generated candidate

`RECIPE_REPLICATION_PREREGISTRATION.md` fixed predictions R1–R5 before the runs. Target:
`L2/cand_14` — a *different* generation with the same skeleton (own approach term with
coefficient 6.0, instantaneous `dock_state`, one-off +300/+30 events), which additionally
returns early while `dock_state` holds, paying **nothing** during the hold; the recipe's
stream is therefore inserted in both branches.

| arm | edit | fresh-60 | dock rate | mean native return |
|---|---|---:|---:|---:|
| `g00_copy` | none (byte-identical) | 0/60 | 0.00 | −1.25 |
| `g01_stream` | recipe (2) only: dense `+20/step` on `dock_state` | 0/60 | 0.00 | −1.25 |
| **`g02_recipe`** | **recipe (1) + (2)** | **23/60 = 38.3 %** | **0.98** | **+124.15** |

All five predictions hold:

* **R1** ✔ `g00_copy` reproduces the original (0/60);
* **R2** ✔ `g02_recipe` = 23/60 = **38.3 %**, far above the pre-registered 8/60 bar — and
  numerically identical to `r09` on the other candidate (23/60);
* **R3** ✔ `g02` (23/60) > `g01` (0/60): the scale half is necessary;
* **R4** ✔ `g01_stream` alone is 0/60 — and its component table is **identical to the copy
  control** (`crate_dock_progress` 100 %, mean generated reward 1.979e−03, `success_event`
  activation 0), i.e. the stream never fires because the predicate is never reached;
* **R5** ✔ `g02`'s `success_event` active rate = **0.0017** (11/20 in-training terminations),
  with realised mass `crate_dock_progress` 43 %, **`success_event` 29 %**,
  `settled_stream` 23 %, `enter_event` 5 %.

**So the two-edit recipe generalises across two independent LLM generations.** This is the
first positive selection-free result of the project: an LLM-written reward, repaired by two
localized edits — one coefficient ×50 and one existing one-off payoff converted into a
per-step stream — delivers the crate at hand-written-arm performance levels.

## 4. What this establishes, and what it does not

**Establishes** (all measured, all reproducible from the JSONs):

* the project's central negative result has a *specific*, two-part cause on this candidate:
  a mis-scaled advantage (so inaction is optimal) plus a payoff structure that pays for
  approaching and not for settling;
* a **two-edit repair reaches 38.3 %**, i.e. the gap between "every LLM reward scores 0 %"
  and "hand-written rewards score 40–98 %" is not a matter of the contract's expressive
  power or of the checks — it is two localized edits inside a reward the generator already
  produced;
* the existing cheap instruments characterise the failure as follows: structural checks see
  nothing, the ordering check is satisfied by a reward that cannot train, the
  component-activity readout is only valid at the full budget, and the advantage-scale probe
  tells you *which* of the two layers you are looking at (`A ≤ 0` ⇒ inaction is optimal)
  without any training at all — but it is not a selector (AUPRC 0.413; it ranks the failing
  L0 candidates highest, because dense farmable terms pay more while acting).

**Does not establish**:

* that EUREKA or CREATE could *find* either edit. Both are small in program space (one
  coefficient; one term converted from one-off to per-step), and the second one is exactly
  the kind of "small enough to be local, large enough that blind search misses it" edit that
  `HANDOFF_QUESTIONS.md` P1–P5 identifies as the theoretical crux — but that is a search
  question this study deliberately does not answer. The measured size of the fix is now
  known, which is what a repair-based follow-up (H2) needed;
* that the recipe covers every LLM failure mode. It was derived on the L2 family (weak,
  gated advantage) and tested on two of its members. The L0 family fails differently — its
  dense terms pay for the wrong behaviour — and the align-fix experiment (§2) shows that
  repairing its identified root cause does **not** make it train, because its advantage stays
  at `A ≈ 0`. Whether a variant of the recipe (e.g. re-scaling the L0 quality term *and*
  gating it on the predicate) fixes that family is untested;
* that any of this is a *method* result. Every arm here was authored by me, not searched.
  The honest framing is: **the target a repair operator must reach is now measured and
  concrete, and it is two edits, not one.**
