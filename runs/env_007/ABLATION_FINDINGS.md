# Where the failure actually is: a single-variable ablation

Date: 2026-09-20.  Environment: `FragileCargoDock-v0`.
All arms: 1.2 M steps, `n_envs=6`, `gamma=0.999`, `normalize_reward=true`,
`reward_clip=20`, seed 0, scored on 20 held-out training-eval seeds and then on
**60 fresh seeds (30000..30059)** that no arm ever trained or selected on.

## The chain of reasoning

1. The LLM search (CREATE v2, EUREKA v2, 30 M steps each) never delivered the
   crate: 0-1 successes in 60 fresh episodes.
2. Two prompt/spec changes - stating the dock geometry in the task spec and
   requiring the completion event to dominate the path - moved the *design*
   metric ~1000x (the dock went from being worth 1.02-1.24x a hover step to
   172-5123x), and produced `cand_03`, the first candidate to ever deliver
   (8.3 % on fresh seeds). It then collapsed to 0 % at 3 M steps.
3. **A hand-written, observation-only control that passes every mechanical check
   scored 6.7 %** - statistically the same as the best LLM candidate. So the
   LLM's reward-design ability is not the binding constraint.
4. Feeding the **native reward through the wrapper** (`return original_reward`)
   scored **19/20 = 95 % at `reward_clip=20`**. So the harness - the wrapper, the
   per-step clip, reward normalisation - is not the binding constraint either.
5. So the gap is inside "what a reward built from observations can express".
   The control and the native reward share their shaping exactly
   (`crate_progress` = decrease of crate-to-dock distance, `cart_approach` =
   decrease of cart-to-crate distance) and differ only in what the native reward
   has on top. Ablating those differences one at a time locates the gap.

## The measurement that points at the mechanism

`diagnose_control.py`, 60 fresh episodes:

| | control v1 (obs-only) | native reward |
|---|---:|---:|
| episodes that enter the dock | 90 % | 90 % |
| steps spent fully inside the dock | **43.5** | 22.1 |
| steps spent inside **and** slower than 0.05 m/s | **4.8** | **9.0** |
| episodes with >= 5 settled steps but never 10 consecutive | **30/60** | 0/60 |

Both get the crate into the dock. The control then spends 43.5 steps inside it
without ever letting the crate come to rest. **The failure is not "cannot get
there", it is "cannot stop".**

## The ablations

Every arm below is the same control v1 with exactly one thing changed.

| arm | change vs control v1 | fresh-60 mean | fresh success | eval-20 |
|---|---|---:|---:|---:|
| control v1 | - | +6.55 | **0/60** | 0/20 |
| **probe A** | `+ roughness` = `-0.02 * peak contact impulse` (from `info`) | +127.63 | **24/60 = 40.0 %** | 11/20 |
| **probe B** | per-step settled stream replaced by the native one-off terminal event (`+5` on dock entry, `+300` on success, `-100` on failure) (from `info`) | +143.87 | **27/60 = 45.0 %** | 4/20 |
| **probe D** | `+ obs-only gentleness proxy` = `-0.05 * contact * max(0, cart_forward_speed - crate velocity along the cart heading)`; **no info fields at all** | **+228.13** | **44/60 = 73.3 %** | 14/20 |
| **probe C** | probe A and probe B together | **+304.39** | **59/60 = 98.3 %** | 17/20 |
| passthrough | the native reward itself, through the wrapper at clip 20 | +278.89 | 54/60 = 90.0 % | 19/20 |
| native reference | the native reward, bypassing the wrapper (calibration, 3 M) | +299.65 | 96.8 % | - |

**Probe C beats the native reward through the same harness (98.3 % vs 90.0 %),
and probe D - which touches no `info` field at all - already reaches 73.3 %.**

### How much each lever is worth

| lever | alone | with the other |
|---|---:|---:|
| gentleness feedback (obs-only proxy) | **73.3 %** | 98.3 % |
| one-off terminal event instead of a per-step stream | 45.0 % | 98.3 % |
| neither | 0 % | - |

The gentleness signal is the larger of the two and, crucially, probe D shows it
does **not** require an environment change: a proxy built from
`obs[14]` (contact), `obs[4]`/`obs[2..3]` (cart speed and heading) and
`obs[8..9]` (crate world velocity) is enough, and outperforms the exact
impulse-based term (73.3 % vs 40.0 %).


**(A) There is no gentle-handling signal.** The environment's reward carries
`-0.02 * peak normal contact impulse` on every contact step. That is the only
term that measures *how hard* the crate is being hit, and it is exactly the
signal needed to learn "release early and let floor drag carry the crate in".
Adding nothing else recovers 40 %.  The observation vector has **no impulse
channel**: `obs[14]` is a contact boolean, `obs[4]` is cart forward speed,
`obs[8..9]` is crate velocity. A reward built only from observations has to
approximate it.

**(B) A per-step reward on the success state is not the same as a terminal
event.** Control v1 pays `+20` for every step the crate is inside, aligned and
slower than 0.05 m/s. The reward for keeping that stream alive competes with
finishing: the episode ends the moment the tenth consecutive settled step is
reached, so "settle for a while, get jostled just above the threshold, settle
again" is a viable policy and is what the control actually does (43.5 steps
inside, 30/60 episodes with >= 5 settled steps but never 10 consecutive).
Replacing the stream with a one-off terminal event recovers 45 %.

Neither cause is in the mechanical checks that were built earlier: control v1
has an *infinite* episode-dominance ratio (a stationary crate anywhere earns
exactly zero) and passes `push > idle` comfortably. Both checks were necessary
and neither was sufficient.

## Consequences

* **The clip is not the problem.** Native through the wrapper at clip 20 is
  95 %; raising the clip to 600 made an LLM candidate *worse*. Leave it at 20.
* **Reward-design prompt rules are not the problem, and more of them have low
  expected value on their own.** A human who knows the whole mechanism and
  writes the reward by hand still lands at 6.7 % without these two ingredients.
* **(B) is fixable in the prompts**, because it is expressible from observations
  alone: a one-off event needs the reward function to keep internal state and to
  detect the episode boundary, which `obs[18]` (fraction of the time budget
  consumed) provides, and the dock-entry event is a plain
  `inside(obs) and not inside(next_obs)` transition.
* **(A) does NOT need an environment change.** Probe D settles it: an
  observation-only proxy recovers more than the exact term does (73.3 % vs
  40.0 %). No new observation dimension is required.
* **The whole gap is therefore addressable in the prompt rules**, and both
  ingredients are mechanically checkable before any training is spent:
  - *gentleness*: probe the reward at "in contact, closing fast" versus
    "in contact, barely closing"; the former must score clearly lower.
  - *terminal form*: call the reward twelve times in a row on the identical
    settled state. A per-step stream accumulates linearly; a one-off event fires
    once. Control v1 fails this; probes B, C and D pass it.

### The size of the result

| policy | fresh-60 success |
|---|---:|
| CREATE v2 best (30 M steps of search) | 1.7 % |
| EUREKA v2 best (30 M steps of search) | 0 % |
| best LLM candidate after the first prompt fix (1.2 M) | 8.3 % |
| hand-written obs-only control (1.2 M / 3 M) | 0 % / 6.7 % |
| **probe C: two prompt-expressible rules applied by hand (1.2 M)** | **98.3 %** |


## Reproduction

```
python diagnose_control.py runs/env_007/control_obs_only/train_3m --episodes 60
python diagnose_control.py runs/env_007/passthrough_probe/clip20 --episodes 60

python -m training.train_sb3_wrapper --config configs/env007_fragilecargo_eureka.yaml \
  --reward runs/env_007/ablation_probe/probeA_plus_roughness.py \
  --save-dir runs/env_007/ablation_probe/probeA \
  --total-timesteps 1200000 --eval-episodes 20 --seed 0

python eval_fresh_seeds.py --episodes 60 --seed-offset 30000 \
  runs/env_007/ablation_probe/probeA
```
