# FragileCargoDock-v0 — PPO baseline calibration

Environment: `FragileCargoDock-v0` (19-D observation, 2-D continuous action, 400-step episodes)
Interpreter: `D:\Code\python\research\llm_env_310\Scripts\python.exe` (Python 3.10.11)
Harness: `run_fragilecargo_baseline.py`, 32 logical cores, Box2D on CPU.

---

## 1. Reference policies

20 episodes, fixed seeds 10000..10019, deterministic actions.

| policy | mean native return | success | dock entered | mean steps |
|---|---:|---:|---:|---:|
| random | −1.14 | 0 % | 0 % | 400 |
| heuristic push controller | 158.41 | 50 % | 90 % | 373 |
| PPO, native reward, converged | **299.65** | **96.8 %** | 97 % | ~176 |

The heuristic controller is the solvability proof; PPO is the reference upper bound.
One successful episode is worth ≈ +310 native return, so a single extra success moves a
20-episode mean by ≈ 15 points.

---

## 2. Hyper-parameter sweep (round 1)

1.2 M timesteps each, 5 envs, seed 0, fixed-seed evaluation every 100 k steps.

| setting | success | mean return | note |
|---|---:|---:|---|
| γ = 0.99 | 0 % | 6.2 | pushes the crate ~1.3 m, never docks |
| γ = 0.995 | 40 % | 126.1 | oscillates 10 % ↔ 60 % |
| γ = 0.999 | 0 % | −0.9 | never even reaches the crate |
| **γ = 0.999 + reward normalisation** | **80 %** | **250.0** | monotone rise |

**Finding:** `normalize_reward` is the decisive knob. At a 400-step horizon with a sparse
+300 terminal bonus, γ = 0.99 discounts that bonus to ≈ 5.4 at t = 0, and the unnormalised
value targets diverge (observed `value_loss` 105 → 167, `explained_variance` → 0.06).
Reward normalisation keeps the value function well conditioned and makes the sparse
terminal signal learnable. High γ alone is not enough — it is actively harmful without it.

---

## 3. Native-reward defects found during calibration

Two mis-specifications in the environment's own (official) reward were found empirically
and corrected. Both are recorded here because they are exactly the kind of diagnostic the
CREATE loop is meant to produce.

### 3.1 Contact bonus created a trapping local optimum

Original term: `gentle_contact = +0.02 * (1 − impulse/threshold)` per step while in contact.

Observed: 2 of 4 seeds converged to a *stuck* policy with native return ≈ **+6.5**.
The crate was pushed ~1.2–1.5 m and jammed against the interior partition (final
crate→dock distance ≈ 2.9 m ≈ the wall position). Leaning on the jammed crate earned
`0.02 × ~250 steps ≈ +5`, which exactly accounted for the plateau.

Fix: replaced the bonus with an impulse-proportional **cost**
(`roughness = −0.02 × peak_impulse`). Paying for mere contact is not a task objective;
"handle gently" is already captured by the hard-impact counter. Leaning now costs slightly
instead of paying.

### 3.2 Removing the bonus destroyed exploration, revealing a missing term

After 3.1, all four seeds sat at ≈ 0 return for 2.25 M steps — 1 of 4 solved, 3 never
touched the crate. The contact bonus had been the only dense bootstrap signal pulling the
cart toward the crate.

Fix: added the `approach_cargo` term from the original design brief — potential-based
shaping on the **cart→crate** distance, weight 1.0. Because it is a telescoping potential
term, its total contribution is bounded by the initial cart–crate distance, so it cannot be
farmed by oscillating.

Effect at 750 k steps (same budget as the failure above):

| | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---:|---:|---:|---:|
| without `approach_cargo` | 0 % | 0 % | 0 % | 0 % |
| with `approach_cargo` | 90 % | 30 % | 80 % | 100 % |

---

## 4. Convergence

Final setting: γ = 0.999, `normalize_reward = true`, `ent_coef = 0.005`, n_envs = 6,
3 M timesteps, seeds 0–3. Fixed-seed evaluation every 250 k steps (10 episodes).

Success rate per checkpoint:

| timesteps | s0 | s1 | s2 | s3 |
|---:|---:|---:|---:|---:|
| 250 k | 0 % | 0 % | 0 % | 0 % |
| 500 k | 20 % | 0 % | 30 % | 0 % |
| 750 k | 90 % | 30 % | 80 % | **100 %** |
| 1000 k | 40 % | 60 % | 90 % | 100 % |
| 1250 k | **100 %** | 90 % | **100 %** | 100 % |
| 1500 k | 100 % | 90 % | 100 % | 100 % |
| 1750 k | 100 % | **100 %** | 100 % | 100 % |
| 2000 k | 80 % | 100 % | 100 % | 90 % |
| 2250 k | 100 % | 90 % | 90 % | 100 % |
| 2500 k | 100 % | 100 % | 100 % | 100 % |
| 2750 k | 100 % | 100 % | 100 % | 100 % |
| 3000 k | 100 % | 100 % | 100 % | 100 % |

First checkpoint from which success stays ≥ 90 %:

| seed | converged at |
|---|---:|
| 3 | 750 k |
| 2 | 1250 k |
| 0 | 1250 k |
| 1 | 1750 k |

**Training budget: 2.5 M steps gives every seed ≥ 90 %; 3.0 M is the recommended budget
with margin.** Below 2 M the spread across seeds is still large (0 %–100 %).

---

## 5. Standard score

100 fresh episodes per seed, seeds 20000..20099, deterministic actions, raw native reward.

| seed | success | mean return | stdev | min | max |
|---|---:|---:|---:|---:|---:|
| 0 | 100 % | 309.65 | 0.35 | 309.08 | 310.59 |
| 1 | 97 % | 300.29 | 52.41 | 0.06 | 310.50 |
| 2 | 97 % | 300.58 | 52.10 | 4.07 | 310.68 |
| 3 | 93 % | 288.08 | 78.62 | −0.39 | 310.55 |
| **pooled (400 ep)** | **96.8 %** | **299.65** | — | — | — |

Native return ≈ `10 + 300 × success_rate`, so:

| success rate | native return |
|---:|---:|
| 0 % (idle) | ≈ −1 |
| 25 % | ≈ 85 |
| 50 % (heuristic) | ≈ 158 |
| 80 % | ≈ 250 |
| 100 % | ≈ 310 |

### Recommended settings for the CREATE config

```yaml
iteration:
  target_score: 250.0            # ≈ 80 % delivery; above the 50 % heuristic, below the 299.65 ceiling
  min_meaningful_improvement: 15.0   # one extra success out of 20 evaluation episodes
training:
  n_envs: 8
  total_timesteps: 3000000
  max_training_steps_for_progress: 3000000
  gamma: 0.999
  normalize_reward: true
  ent_coef: 0.005
```

`target_score = 250` is the recommended "solved" bar. The converged reference score is
≈ 300; setting the target at the ceiling would leave no headroom, and setting it near 158
would accept the heuristic controller's 50 % delivery rate.

---

## 6. Reproduction

```powershell
$py = "D:\Code\python\research\llm_env_310\Scripts\python.exe"
cd D:\Code\python\research\form_github\expert-reward-agent

# reference policies
& $py run_fragilecargo_baseline.py --policies random,heuristic --episodes 20

# one calibrated training run
& $py run_fragilecargo_baseline.py --policies ppo --total-timesteps 3000000 `
    --n-envs 6 --eval-every 250000 --gamma 0.999 --normalize-reward `
    --seed 0 --out-dir runs/env_007/ac3_s0

# re-evaluate a saved model on fresh seeds
& $py run_fragilecargo_baseline.py --policies ppo --episodes 100 --eval-seed-offset 20000 `
    --load-model runs/env_007/ac3_s0/model.zip `
    --load-vecnormalize runs/env_007/ac3_s0/vecnormalize.pkl `
    --out-dir runs/env_007/confirm_s0
```
