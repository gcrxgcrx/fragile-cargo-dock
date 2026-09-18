# FragileCargoDock-v0 baseline

- env id: `FragileCargoDock-v0`
- evaluation: 30 episodes, seeds 10000..10029
- PPO: 3,000,000 timesteps, n_envs=6, seed=0, gamma=0.999, ent_coef=0.005, arch=128,128

| policy | mean return | stdev | success | dock entered | mean steps | final goal dist (m) | final angle err (rad) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppo_native_reward | 299.62 | 54.09 | 96.7% | 100.0% | 187.2 | 0.032 | 0.193 |

## Learning curve (fixed-seed evaluation during training)

| timesteps | mean return | success | dock entered | goal dist (m) |
|---:|---:|---:|---:|---:|
| 250,002 | 3.38 | 0.0% | 20.0% | 1.541 |
| 500,004 | 65.99 | 20.0% | 40.0% | 0.348 |
| 750,000 | 279.50 | 90.0% | 100.0% | 0.092 |
| 1,000,002 | 129.21 | 40.0% | 100.0% | 0.174 |
| 1,250,004 | 309.53 | 100.0% | 100.0% | 0.069 |
| 1,500,000 | 309.47 | 100.0% | 100.0% | 0.059 |
| 1,750,002 | 309.39 | 100.0% | 100.0% | 0.038 |
| 2,000,004 | 248.32 | 80.0% | 80.0% | 0.152 |
| 2,250,000 | 309.38 | 100.0% | 100.0% | 0.023 |
| 2,500,002 | 309.52 | 100.0% | 100.0% | 0.044 |
| 2,750,004 | 309.61 | 100.0% | 100.0% | 0.037 |
| 3,000,000 | 309.50 | 100.0% | 100.0% | 0.039 |

## Convergence

- best success rate observed: 100.0%
- best mean return observed: 309.61
- first checkpoint at/above half of best success: 750000
- last checkpoint below half of best success: 1000002

## Notes

- `ppo_native_reward` trains on the environment's own native reward, i.e. the
  `--use-original-reward` reference used to bound what a generated reward can
  realistically reach.
- One successful episode contributes about +311 to that episode's native
  return, so each additional success shifts `mean_return` by roughly
  311/episodes.
