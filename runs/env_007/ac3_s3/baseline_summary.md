# FragileCargoDock-v0 baseline

- env id: `FragileCargoDock-v0`
- evaluation: 30 episodes, seeds 10000..10029
- PPO: 3,000,000 timesteps, n_envs=6, seed=3, gamma=0.999, ent_coef=0.005, arch=128,128

| policy | mean return | stdev | success | dock entered | mean steps | final goal dist (m) | final angle err (rad) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppo_native_reward | 309.69 | 0.30 | 100.0% | 100.0% | 168.5 | 0.037 | 0.066 |

## Learning curve (fixed-seed evaluation during training)

| timesteps | mean return | success | dock entered | goal dist (m) |
|---:|---:|---:|---:|---:|
| 250,002 | 3.10 | 0.0% | 30.0% | 2.267 |
| 500,004 | 6.98 | 0.0% | 80.0% | 1.169 |
| 750,000 | 309.62 | 100.0% | 100.0% | 0.067 |
| 1,000,002 | 309.59 | 100.0% | 100.0% | 0.088 |
| 1,250,004 | 309.52 | 100.0% | 100.0% | 0.073 |
| 1,500,000 | 309.56 | 100.0% | 100.0% | 0.058 |
| 1,750,002 | 309.54 | 100.0% | 100.0% | 0.062 |
| 2,000,004 | 279.38 | 90.0% | 100.0% | 0.077 |
| 2,250,000 | 309.51 | 100.0% | 100.0% | 0.048 |
| 2,500,002 | 309.45 | 100.0% | 100.0% | 0.061 |
| 2,750,004 | 309.44 | 100.0% | 100.0% | 0.055 |
| 3,000,000 | 309.56 | 100.0% | 100.0% | 0.039 |

## Convergence

- best success rate observed: 100.0%
- best mean return observed: 309.62
- first checkpoint at/above half of best success: 750000
- last checkpoint below half of best success: 500004

## Notes

- `ppo_native_reward` trains on the environment's own native reward, i.e. the
  `--use-original-reward` reference used to bound what a generated reward can
  realistically reach.
- One successful episode contributes about +311 to that episode's native
  return, so each additional success shifts `mean_return` by roughly
  311/episodes.
