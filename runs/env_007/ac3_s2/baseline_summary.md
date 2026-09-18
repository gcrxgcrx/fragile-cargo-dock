# FragileCargoDock-v0 baseline

- env id: `FragileCargoDock-v0`
- evaluation: 30 episodes, seeds 10000..10029
- PPO: 3,000,000 timesteps, n_envs=6, seed=2, gamma=0.999, ent_coef=0.005, arch=128,128

| policy | mean return | stdev | success | dock entered | mean steps | final goal dist (m) | final angle err (rad) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppo_native_reward | 309.75 | 0.32 | 100.0% | 100.0% | 162.8 | 0.063 | 0.178 |

## Learning curve (fixed-seed evaluation during training)

| timesteps | mean return | success | dock entered | goal dist (m) |
|---:|---:|---:|---:|---:|
| 250,002 | 4.04 | 0.0% | 10.0% | 0.774 |
| 500,004 | 98.32 | 30.0% | 90.0% | 0.544 |
| 750,000 | 248.75 | 80.0% | 90.0% | 0.289 |
| 1,000,002 | 279.51 | 90.0% | 100.0% | 0.086 |
| 1,250,004 | 309.59 | 100.0% | 100.0% | 0.060 |
| 1,500,000 | 309.61 | 100.0% | 100.0% | 0.072 |
| 1,750,002 | 309.60 | 100.0% | 100.0% | 0.073 |
| 2,000,004 | 309.62 | 100.0% | 100.0% | 0.033 |
| 2,250,000 | 279.09 | 90.0% | 90.0% | 0.072 |
| 2,500,002 | 309.68 | 100.0% | 100.0% | 0.054 |
| 2,750,004 | 309.62 | 100.0% | 100.0% | 0.057 |
| 3,000,000 | 309.63 | 100.0% | 100.0% | 0.057 |

## Convergence

- best success rate observed: 100.0%
- best mean return observed: 309.68
- first checkpoint at/above half of best success: 750000
- last checkpoint below half of best success: 500004

## Notes

- `ppo_native_reward` trains on the environment's own native reward, i.e. the
  `--use-original-reward` reference used to bound what a generated reward can
  realistically reach.
- One successful episode contributes about +311 to that episode's native
  return, so each additional success shifts `mean_return` by roughly
  311/episodes.
