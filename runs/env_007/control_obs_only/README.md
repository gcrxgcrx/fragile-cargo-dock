# Control: what is the ceiling of an observation-only reward?

This directory holds a hand-written reward (`reward.py`) and its two training
runs. It exists to answer a question the project has never tested:

> Can **any** reward that only reads the observation vector reach a high
> delivery rate under this harness
> (`reward_clip = 20.0`, `normalize_reward = true`, `gamma = 0.999`,
> `n_envs = 6`, 3M steps)?

The 96.8 % / 299.65 reference in `runs/env_007/CALIBRATION.md` is the
**native** reward. The native reward bypasses `RewardOverrideWrapper`
entirely - it is never clipped and never normalised - so it says nothing about
what the generated-reward contract can express. If this control also fails,
then no prompt engineering can fix the search, because the bottleneck would be
the harness, not the generator.

## The control reward

Hand-written to satisfy every mechanical check in `analyze_terminal_dominance.py`:

| check | result |
|---|---|
| reward at the dock, per step | +20.00 (the clip maximum) |
| reward parked at 0.30 m / 0.13 m / inside-but-moving / mid-transit | **exactly 0.00** |
| episode-credit ratio, `10*20 / (400*0)` | **infinite -> PASS** |
| `idle` vs `push_forever` vs `expert` cumulative | 0.00 < 4.76 < 64.65 -> **PASS** |

Design constraints it obeys:

* every shaping term is an increment `f(obs) - f(next_obs)`, so nothing can be
  farmed by standing still;
* the completion condition is the task spec's, verbatim:
  `|obs[12]| <= 0.024`, `|obs[13]| <= 0.030`, `obs[10] >= cos 30 deg`,
  crate speed < 0.05 m/s;
* the completion term pays the clip maximum for every one of the ten settling
  steps, so a completion is worth 200 against a dense path worth ~4.1 (49:1),
  comparable to the native reward's 300 against ~10 (30:1);
* the only negative terms are a boundary guard and a "violent shove" guard, both
  inactive during normal pushing, so ending the episode early is never
  attractive - which is what drove `cand_00` and `cand_02` out of bounds.

## Decision rule (fixed before the runs finish)

| 1.2M fresh-60 | 3.0M fresh-60 | conclusion |
|---|---|---|
| >= 50 % | >= 50 % | The contract can express a solvable reward. The search's problem is design quality, the mechanical gate is the right target, and prompt work is worth doing. |
| 5 - 50 % | any | The contract is workable but much harder than the native one. The gate needs a difficulty term, not just a dominance term. |
| ~0 % | ~0 % | **The bottleneck is the harness, not the LLM.** `reward_clip = 20` versus a native completion bonus of 300 is then the prime suspect, and the fix is a harness change (clip / normalisation / gamma), not a prompt change. |

Reference points at the same budget:

| policy | budget | fresh-60 | success |
|---|---:|---:|---:|
| pilot cand_03 (LLM, new rules) | 1.2M | +31.30 | 5/60 = 8.3 % |
| pilot cand_03 (same reward) | 3.0M | +4.00 | 0/60 |
| native-reward PPO | 3.0M | +299.65 | 96.8 % |
| CREATE v2 best | 3.0M | +7.72 | 1/60 |

## Result (measured)

Fresh seeds 30000..30059, 60 episodes, seeds the search never saw:

| policy | budget | fresh-60 | success | dock entered | crate speed at dock entry | mean best settled steps |
|---|---:|---:|---:|---:|---:|---:|
| native reward, **bypassing the wrapper** | 3.0M | +299.65 | 96.8 % | - | - | - |
| native reward | 1.2M | - | 80 % | - | - | - |
| **this control** | 3.0M | +27.47 | **4/60 = 6.7 %** | 54/60 = 90 % | **0.243 m/s** | **4.23 / 10** |
| **this control** | 1.2M | +6.55 | 0/60 | 41/60 = 68 % | - | - |
| best LLM candidate (cand_03) | 1.2M | +31.30 | 5/60 = 8.3 % | 26/60 = 43 % | 0.185 m/s | 0.93 / 10 |
| best LLM candidate (cand_03) | 3.0M | +4.00 | 0/60 | 0/60 | - | - |

Distribution of the best run of consecutive settled steps, over 60 episodes
(10 is required to succeed):

```
control :  0 -> 11    1-4 -> 23    5-9 -> 22    10 -> 4
cand_03 :  0 -> 53    1-4 ->  1    5-9 ->  1    10 -> 5
```

**The answer to the question this control was built to ask is "no": the
observation-only contract is not the binding constraint on the generator.**
A hand-written reward that satisfies every mechanical check performs the same as
the best reward the search ever produced (6.7 % vs 8.3 %, and the difference is
within noise at n = 60), against the native reward's 96.8 %. Since a human
writing the reward by hand with full knowledge of the mechanics cannot do
better, further prompt engineering has low expected value.

The mechanism is the same in both: the crate is still doing **0.19 - 0.24 m/s**
at the moment it first sits fully inside the dock, and success needs
< 0.05 m/s. Neither policy learns "release early and let floor drag carry the
crate in". The control gets 22/60 episodes to 5-9 of the 10 required settling
steps - it learns the approach and then cannot hold the last second.

## What is left as the suspect

The native reward is measured with `reward_fn = None`, which bypasses
`RewardOverrideWrapper` completely - it is never clipped. Every generated reward
goes through the wrapper, which clips the per-step reward to
`reward_clip = 20.0`, while the native completion term is **+300**.

`../passthrough_probe/` tests exactly this, by feeding the native reward through
the wrapper at clip 20 and clip 600.

## Reproduction

```
python analyze_terminal_dominance.py --clip 20 --episodes 6 runs/env_007/control_obs_only/reward.py

python -m training.train_sb3_wrapper --config configs/env007_fragilecargo_eureka.yaml \
  --reward runs/env_007/control_obs_only/reward.py \
  --save-dir runs/env_007/control_obs_only/train_3m \
  --total-timesteps 3000000 --eval-episodes 20 --seed 0

python eval_fresh_seeds.py --episodes 60 --seed-offset 30000 \
  runs/env_007/control_obs_only/train_3m
```
