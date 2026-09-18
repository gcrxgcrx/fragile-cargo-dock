# FragileCargoDock-v0 baseline

- env id: `FragileCargoDock-v0`
- evaluation: 30 episodes, seeds 10000..10029
- PPO: 3,000,000 timesteps, n_envs=6, seed=1, gamma=0.999, ent_coef=0.005, arch=128,128

| policy | mean return | stdev | success | dock entered | mean steps | final goal dist (m) | final angle err (rad) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppo_native_reward | 309.50 | 0.31 | 100.0% | 100.0% | 184.4 | 0.052 | 0.332 |

## Learning curve (fixed-seed evaluation during training)

| timesteps | mean return | success | dock entered | goal dist (m) |
|---:|---:|---:|---:|---:|
| 250,002 | 4.40 | 0.0% | 30.0% | 1.330 |
| 500,004 | 2.91 | 0.0% | 20.0% | 1.732 |
| 750,000 | 95.47 | 30.0% | 30.0% | 0.178 |
| 1,000,002 | 187.15 | 60.0% | 60.0% | 0.112 |
| 1,250,004 | 279.28 | 90.0% | 100.0% | 0.061 |
| 1,500,000 | 278.69 | 90.0% | 90.0% | 0.105 |
| 1,750,002 | 309.39 | 100.0% | 100.0% | 0.074 |
| 2,000,004 | 309.52 | 100.0% | 100.0% | 0.036 |
| 2,250,000 | 278.75 | 90.0% | 90.0% | 0.143 |
| 2,500,002 | 309.43 | 100.0% | 100.0% | 0.063 |
| 2,750,004 | 309.37 | 100.0% | 100.0% | 0.080 |
| 3,000,000 | 309.39 | 100.0% | 100.0% | 0.048 |

## Convergence

- best success rate observed: 100.0%
- best mean return observed: 309.52
- first checkpoint at/above half of best success: 1000002
- last checkpoint below half of best success: 750000

## Notes

- `ppo_native_reward` trains on the environment's own native reward, i.e. the
  `--use-original-reward` reference used to bound what a generated reward can
  realistically reach.
- One successful episode contributes about +311 to that episode's native
  return, so each additional success shifts `mean_return` by roughly
  311/episodes.
