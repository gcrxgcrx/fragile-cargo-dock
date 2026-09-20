# Pilot: "terminal event must dominate the path" + exact dock geometry

Run date: 2026-09-20.  Everything here runs on `FragileCargoDock-v0`.

## Question

The reward search had produced ten generations of candidates and none of them
ever delivered the crate (0 % success on fresh seeds, best fresh-60 score
+7.7 with 1/60). The diagnosis was that every generated reward is
*path-shaped*: a dense term keeps paying while the crate sits near the dock,
and no candidate gives the one-off completion event enough weight to beat it.

Two changes were proposed and probed in isolation:

1. **task spec** (`envs/env_007/task_spec_anonymized_v2.yaml`) - state the dock
   geometry, so that "fully inside the dock" becomes a *computable* predicate
   instead of a guess:
   `|obs[12]| <= 0.024 and |obs[13]| <= 0.030`.
   Before this the card said only "the crate must be fully inside the dock
   rectangle" and gave no size, so a generator could not write the predicate.
2. **generator prompt** (`prompts/eureka_01_initial_reward_v2.md`) - add a hard
   rule: the per-step reward for satisfying the completion condition, `B`, must
   satisfy `10 * B > 3 * (path_bound * 400)`, plus self-check (2)
   "hovering 0.30 m from the dock must score below actually docking".

Nothing that was already running was touched: the probe uses its own config,
prefix, run root, task spec and prompt copies (`prompts/` is re-read from disk
every round, so editing the live files would have contaminated an in-flight
CREATE round).

```
python -m pipeline.run_eureka_population \
  --config configs/env007_terminal_rule_pilot.yaml \
  --prefix terminal_rule_pilot --generations 1 \
  --population-size 4 --parallel-candidates 4 --total-timesteps 1200000
```

## Result 1 - the rule was understood and applied

`analyze_terminal_dominance.py` scores each reward on four scripted states.
All four generation-0 candidates came back structurally correct:

| candidate | reward, one docked step | reward, hovering 0.30 m, at rest | dense payoff per step |
|---|---:|---:|---:|
| cand_00 | +501 | +0.70 | hover pays |
| cand_01 | +2003 | +0.00 | nothing outside the dock |
| cand_02 | +251 | +1.43 | hover pays |
| cand_03 | +602 | +1.38 | hover pays |

Every one of them derives the predicate from the observation using the exact
tolerance from the new spec. `cand_03` even writes the rule's arithmetic into a
comment:

```python
# 过程上限估计：12*0.05 + 2 + 3*~0.5 ≈ 0.6+2+1.5 = 4.1 每步，*400 = 1640
# 10*B > 3*1640 => B > 492，取 600 留余量
dock_event = 600.0
```

Compare with the previous generation-0 population (EUREKA v2, no rule), where
the best candidate used a plain proximity proxy and never expressed a
completion event at all.

The scripted-controller check (`idle` / `push_forever` / `expert`) also passes
for all four: `expert > push_forever > idle`, with `push_forever` scoring
*negative* for three of them - i.e. the reward no longer rewards shoving, but
does reward the release-and-settle manoeuvre.

## Result 2 - the per-step clip is a red herring (and raising it hurts)

`RewardOverrideWrapper` clips the generated reward to `reward_clip`, default
**20.0** (`training/train_sb3_wrapper.py:656`). `cand_00`'s +501 docked step is
therefore squashed to +20, i.e. at most 10 x 20 = 200 over the settling window,
against 0.70 x 400 = 280 of pure hover payoff. That looked like the real
blocker, so the same four reward files were retrained with `reward_clip: 600`
(`configs/env007_terminal_rule_pilot_clip600.yaml`) - a controlled A/B in which
*only* the clip differs.

| candidate | clip = 20 (default) | clip = 600 |
|---|---|---|
| cand_00 | -20.517, 0/20, max +3.18 | -41.068, 0/20, max +1.03 |
| cand_01 | -1.730, 0/20 | -1.730, 0/20 (bit-identical) |
| cand_02 | -102.998, 0/20 (20/20 out of bounds) | -102.998, 0/20 (bit-identical) |
| cand_03 | **+34.884, 2/20, max +309.64** | +2.886, 0/20, max +9.60 |

Raising the clip **made the only working candidate worse**. The interpretation:
with `normalize_reward=true`, a 600-per-step terminal spike inflates the return
standard deviation, so the dense shaping that actually steers the crate to the
dock is scaled into irrelevance, while the sparse event is too rare to learn
from. Under clip=600 `cand_03` reaches the dock *more* often (52/60 vs 26/60,
final distance 0.175 m vs 0.311 m) but never satisfies the low-speed condition.

`cand_01` and `cand_02` scoring bit-identically in both arms is a useful
control: their per-step rewards never exceeded 20 during the rollouts, so the
clip never bound - which also means `cand_01` never reached its own +2003
completion state.

Conclusion: keep `reward_clip = 20.0`. The terminal-dominance rule is still the
right rule at the *design* level, but "dominate" has to be read relative to the
reward's own scale, not by inflating `B` without bound.

## Result 3 - the first generated reward that actually delivers

Fresh seeds 30000..30059, 60 episodes, seeds the search never evaluated on:

| policy | mean native return | success | dock entered | best final distance |
|---|---:|---:|---:|---:|
| native-reward PPO (calibration) | 299.65 | 96.8 % | - | - |
| heuristic controller | 158.41 | 50 % | - | - |
| **cand_03, 1.2M steps, clip 20** | **+31.30** | **5/60 = 8.3 %** | 26/60 | 0.106 m |
| **cand_03, 3.0M steps, clip 20** | **+4.00** | **0/60** | **0/60** | 0.173 m |
| cand_03, 1.2M steps, clip 600 | +6.88 | 0/60 | 52/60 | 0.013 m |
| cand_00, 1.2M steps, clip 20 | -39.62 | 0/60 | 0/60 | 0.663 m |

For reference, the previous best results from the main experiments, both at
2.5x the training budget:

| run | budget | selection score (seeds 10000-10019) | fresh-60 score | fresh success |
|---|---|---:|---:|---:|
| CREATE v2 best (iter_04) | 3.0M | +17.694 | +7.716 | 1/60 = 1.7 % |
| EUREKA v2 best (g02c03) | 3.0M | +8.093 | +7.027 | 0/60 |
| **pilot cand_03** | **1.2M** | +34.884 | **+31.30** | **5/60 = 8.3 %** |

## Result 4 - the four candidates fail in four different ways

Reading the reward files against the scripted-controller numbers explains each
result, and points at what the next prompt revision has to fix.

The one test that separates them is `push_forever` vs `idle`, i.e. "is pushing
the crate toward the dock worth more than doing nothing":

| candidate | idle | push_forever | expert | passes? | outcome |
|---|---:|---:|---:|---|---|
| cand_00 | +0.00 | +28.82 | +100.05 | yes | dies out of bounds, 23/60 |
| cand_01 | +0.00 | **-7.98** | +6017 | **no, idle wins** | never moves the crate, 0/60 |
| cand_02 | -1.99 | +155.34 | +916.05 | yes | dies out of bounds, 20/20 |
| cand_03 | +0.00 | +51.93 | +1880.39 | yes | **delivers, 5/60** |

**cand_01 - the pushing penalty outweighs the pushing reward.** It has the
cleanest structure of the four (no hover payoff at all, a pure incremental
`6.0 * progress_delta`), yet it never moves the crate. Its collision guard

```python
if contact > 0.5:
    contact_risk = -0.5 * min(1.0, crate_speed / 1.0) * min(1.0, cart_speed / 1.5)
```

charges about -0.12 per step while pushing, while the progress term pays only
about +0.10 per step, and `-0.05 * (a0^2 + a1^2)` takes another -0.05. Net
pushing is negative, so standing still is optimal. This is exactly the
incentive conflict the existing rule forbids, and the existing self-check ("(2)
must beat (1) on a single step") would have caught it - `cand_01` simply did
not evaluate its own check.

**cand_00 and cand_02 - the out-of-bounds guard is mis-scaled.** The spec says
the observation is *clipped to [-2, 2]*, and both candidates took that to be the
physical range:

* `cand_00` guards at `|obs[0]| > 1.8`, which corresponds to 9.0 m. The cart
  actually leaves the field at 5.25 m, i.e. `|obs[0]| > 1.05`. The guard can
  never fire.
* `cand_02` guards at 0.85 but only charges `-0.5 * (|x| - 0.85)`, i.e. about
  -0.1 per step at the boundary, against a `-100` native failure.

The observation field definition does say "divided by warehouse half-width
(0 = centre line, +1 = far wall)", so the information is there - but the
"clipped to [-2.0, 2.0]" line invites the wrong inference.

There is a second, subtler problem behind cand_02's 20/20 out-of-bounds
failures: **the generated reward has no failure signal at all.** `info` is
declared unusable, so a candidate cannot know the episode ended badly. When the
dense terms are net negative (which they are while the crate is being pushed
away from the dock), ending the episode early is *rewarding*, and driving off
the field is the cheapest way to do it. `cand_02` ends after 125 steps instead
of 400.

**cand_03 works** because it is the only one that got three things right at
once: a correctly scaled boundary guard (`|obs[0]| > 0.95`, weight 5.0), a
capped incremental progress term that is positive on net, and the completion
event. It also carries the largest hover payoff of the four, which is the price
it pays for being clipped at 20.

### Consequences for the next prompt revision

1. Keep the terminal-dominance rule - it is what produced `cand_03`.
2. Add an explicit statement of the physical range of `obs[0]`/`obs[1]`
   (`+/-1.0` is the wall, the cart leaves the field past `1.05`), and say that
   the `[-2, 2]` in the observation space is only a global clip.
3. State that the generated reward receives **no** terminal failure penalty,
   so the reward itself has to make going out of bounds clearly worse than
   staying in - otherwise ending the episode early is attractive whenever the
   dense terms are negative.
4. Extend `analyze_terminal_dominance.py` with an excursion test: the reward of
   a scripted "drive straight off the field" rollout must be clearly below the
   idle rollout.

## Result 5 - the 8.3 % was a transient, and a one-line check predicts it

Retraining `cand_03`'s reward for the full 3.0M steps **destroys it**:

| budget | selection score (10000-10019) | fresh-60 | fresh success | fresh dock entries |
|---|---:|---:|---:|---:|
| 1.2M | +34.884 (2/20) | +31.30 | 5/60 = 8.3 % | 26/60 |
| 3.0M | +4.152 (0/20) | +4.00 | 0/60 | **0/60** |

During the long run the *generated* return climbs monotonically (100 -> 126
per episode) while the native success rate stays at 0-2 %. The policy is
farming the generated reward: it parks the crate just outside the dock and
collects `crate_align = 2.0 * near_gate * align_score` forever, giving up on
docking altogether - final distance settles at 0.32 m, and the dock is never
entered once in 60 episodes. 1.2M steps was simply the moment before the hack
took over.

This is exactly what the single-step probe cannot see. Adding the horizon and
the clip to the probe makes the failure *predictable before training*:

```
completion credit = STABLE_STEPS_REQUIRED x clipped(reward at the dock)   = 10 * 20 = 200
hover credit      = EPISODE_STEPS        x clipped(best legal non-completion state)
```

The worst legal non-completion states are "parked just outside the tolerance"
(`|obs[12]|` slightly above 0.024) and "inside the dock but still moving", both
of which pay every step and never end the episode.

| candidate | completion credit | best hover credit | ratio | verdict | measured outcome |
|---|---:|---:|---:|---|---|
| cand_00 | 200 | 348 | 0.58 | FAIL | 0/60, dies out of bounds |
| cand_01 | 200 | 157 | 1.28 | marginal | 0/60, never moves the crate |
| cand_02 | 200 | 585 | 0.34 | FAIL | 0/20, dies out of bounds |
| **cand_03** | 200 | **692** | **0.29** | **FAIL - worst of the four** | **the one that visibly collapsed** |

**No candidate passes.** The one with the *worst* ratio is the one that
appeared to work at 1.2M and then collapsed at 3M - the check ranks them in the
right order, and it costs seconds instead of 25 minutes of training.

The root cause is now precise, and it is not "the terminal bonus is too
small": it is that **the near-goal terms are persistent state rewards rather
than increments**. A stationary crate at 0.13 m collects 1.73 per step forever,
so no finite one-off bonus can win against it over a 400-step episode unless
the bonus also survives the clip - and making the bonus survive the clip (600)
was already shown to break training through return normalisation.

## Result 6 - what the next prompt revision has to say

Two independent defects, both now mechanically checkable:

1. **Terminal dominance must be stated in episode credit, not per step.**
   `STABLE_STEPS_REQUIRED * min(B, clip) > 3 * EPISODE_STEPS * (best legal
   non-completion step reward)`. The current rule compares a single step of `B`
   against `path_bound * 400` but never bounds the *hover* state, so all four
   candidates satisfied the letter of the rule and three of them fail it.
2. **Near-goal auxiliary terms must be increments, not states.** The existing
   rule says a "slow / still / aligned" quantity may only be a *gate multiplied
   onto the progress signal*; all four candidates instead multiplied it onto a
   static state, which is persistent by construction. That must be spelled out
   with the arithmetic: "a stationary crate at 0.13 m must score exactly 0 per
   step".

Both checks belong in `analyze_terminal_dominance.py` as a **pre-training
gate**: a candidate whose ratio is <= 1 should not be trained at all, let alone
be selected.

## Result 7 - all 24 candidates on one ruler

`analyze_terminal_dominance.py --clip 20 --summary` run over every reward the
project has produced: 10 EUREKA v2 candidates (4 generations), 10 CREATE v2
candidates (10 rounds), and the 4 pilot candidates. No training involved.

Two quantities per candidate:

* **single-step ratio** = `reward at the dock / reward parked just outside`
  (`|obs[12]|` slightly above the tolerance). Clip-independent, pure design.
* **episode-credit ratio** = `10 * clipped(docked) / (400 * clipped(hover))`.
  What the policy actually sees, including the per-step clip and the horizon.

| contract | n | docked step, raw | single-step ratio | episode-credit ratio |
|---|---:|---|---:|---:|
| EUREKA v2 + CREATE v2 (old spec, old prompt) | 20 | 0.00 - **20.00** | **1.02 - 1.24** | **0.025 - 0.031** |
| pilot (dock geometry + terminal rule) | 4 | 251 - **2003** | **172 - 5123** | **0.289 - 1.278** |

Read the old block again: every one of the twenty candidates pays the *same*
reward for being docked as for hovering 0.13 m away - the ratio is 1.02 to 1.24,
i.e. the dock is worth 2-24 % more than not docking. And the largest single-step
value any of them ever assigned to the dock is exactly **20.00**, the clip
value: the old search never even *tried* to signal a completion event.

The new block assigns the dock 172x to 5123x a hover step. That is a ~1000x
change in the design, measured on every candidate, and it does not depend on any
training result or on n = 4.

The residual problem is visible in the same table: with `reward_clip = 20` and a
400-step episode, `10 * 20 = 200` still loses to `400 * (0.39 .. 1.73) = 157 ..
692`. The design is fixed; the *bookkeeping* is not. Making the hover states pay
exactly zero would make the episode-credit ratio unbounded regardless of the
clip - that is the hypothesis that has not been tested yet.

## Caveats

* n = 4 candidates for the new prompt. This is a strong signal, not a
  convergence result: one candidate of four works.
* Only the *initial* generator prompt was changed. `prompts/02_reward_generator_prompt.md`,
  `prompts/reflection_agent_prompt.md`, `prompts/eureka_02_reward_edit.md` and
  the live `envs/env_007/task_spec_anonymized.yaml` are untouched, so the
  main CREATE and EUREKA runs were produced under the old contract and are not
  comparable to this pilot.
* The 1.2M-step budget is 40 % of the main runs, so `cand_03` is not yet
  converged; its ceiling is unknown.

## Reproduction

```
python -m pipeline.run_eureka_population \
  --config configs/env007_terminal_rule_pilot.yaml \
  --prefix terminal_rule_pilot --generations 1 --population-size 4 \
  --parallel-candidates 4 --total-timesteps 1200000 --eval-episodes 20

# mechanical structure check, no training required
python analyze_terminal_dominance.py --clip 20 --episodes 6 \
  runs/env_007/terminal_rule_pilot/seed_0/gen_00/cand_*/reward_v1.py

# clip A/B
python pilot_train_existing.py --config configs/env007_terminal_rule_pilot_clip600.yaml \
  --src runs/env_007/terminal_rule_pilot/seed_0/gen_00 \
  --dst runs/env_007/terminal_rule_pilot_clip600/seed_0/gen_00 \
  --total-timesteps 1200000 --eval-episodes 20 --parallel 4

# honest scoring on unseen seeds
python eval_fresh_seeds.py --episodes 60 --seed-offset 30000 \
  runs/env_007/terminal_rule_pilot/seed_0/gen_00/cand_03/training
```
