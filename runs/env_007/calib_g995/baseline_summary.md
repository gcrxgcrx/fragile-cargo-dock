# FragileCargoDock-v0 baseline

- env id: `FragileCargoDock-v0`
- evaluation: 20 episodes, seeds 10000..10019
- PPO: 1,200,000 timesteps, n_envs=5, seed=0, gamma=0.995, ent_coef=0.005, arch=128,128

| policy | mean return | stdev | success | dock entered | mean steps | final goal dist (m) | final angle err (rad) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppo_native_reward | 126.10 | 150.32 | 40.0% | 40.0% | 305.1 | 0.772 | 0.667 |

## Learning curve (fixed-seed evaluation during training)

| timesteps | mean return | success | dock entered | goal dist (m) |
|---:|---:|---:|---:|---:|
| 100,000 | 0.86 | 0.0% | 0.0% | 3.790 |
| 200,000 | 0.41 | 0.0% | 0.0% | 3.426 |
| 300,000 | 2.98 | 0.0% | 0.0% | 3.096 |
| 400,000 | 3.87 | 0.0% | 0.0% | 2.052 |
| 500,000 | 3.19 | 0.0% | 10.0% | 2.500 |
| 600,000 | 6.81 | 0.0% | 50.0% | 1.238 |
| 700,000 | 65.85 | 20.0% | 40.0% | 1.076 |
| 800,000 | 35.01 | 10.0% | 20.0% | 0.970 |
| 900,000 | 189.06 | 60.0% | 90.0% | 0.461 |
| 1,000,000 | 34.92 | 10.0% | 10.0% | 0.886 |
| 1,100,000 | 96.28 | 30.0% | 40.0% | 0.638 |
| 1,200,000 | 126.36 | 40.0% | 40.0% | 0.498 |

## Convergence

- best success rate observed: 60.0%
- best mean return observed: 189.06
- first checkpoint at/above half of best success: 900000
- last checkpoint below half of best success: 1000000

## Notes

- `ppo_native_reward` trains on the environment's own native reward, i.e. the
  `--use-original-reward` reference used to bound what a generated reward can
  realistically reach.
- One successful episode contributes about +311 to that episode's native
  return, so each additional success shifts `mean_return` by roughly
  311/episodes.
