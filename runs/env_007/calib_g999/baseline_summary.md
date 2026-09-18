# FragileCargoDock-v0 baseline

- env id: `FragileCargoDock-v0`
- evaluation: 20 episodes, seeds 10000..10019
- PPO: 1,200,000 timesteps, n_envs=5, seed=0, gamma=0.999, ent_coef=0.005, arch=128,128

| policy | mean return | stdev | success | dock entered | mean steps | final goal dist (m) | final angle err (rad) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppo_native_reward | -0.85 | 0.07 | 0.0% | 0.0% | 400.0 | 4.128 | 0.157 |

## Learning curve (fixed-seed evaluation during training)

| timesteps | mean return | success | dock entered | goal dist (m) |
|---:|---:|---:|---:|---:|
| 100,000 | -8.86 | 0.0% | 0.0% | 3.512 |
| 200,000 | 0.85 | 0.0% | 0.0% | 3.833 |
| 300,000 | -0.19 | 0.0% | 0.0% | 3.882 |
| 400,000 | -0.82 | 0.0% | 0.0% | 4.071 |
| 500,000 | -0.46 | 0.0% | 0.0% | 3.900 |
| 600,000 | -0.79 | 0.0% | 0.0% | 4.049 |
| 700,000 | -0.77 | 0.0% | 0.0% | 4.054 |
| 800,000 | -0.83 | 0.0% | 0.0% | 4.070 |
| 900,000 | -0.80 | 0.0% | 0.0% | 4.052 |
| 1,000,000 | -0.83 | 0.0% | 0.0% | 4.070 |
| 1,100,000 | -0.83 | 0.0% | 0.0% | 4.070 |
| 1,200,000 | -0.84 | 0.0% | 0.0% | 4.070 |

## Convergence

- best success rate observed: 0.0%
- best mean return observed: 0.85
- first checkpoint at/above half of best success: None
- last checkpoint below half of best success: None

## Notes

- `ppo_native_reward` trains on the environment's own native reward, i.e. the
  `--use-original-reward` reference used to bound what a generated reward can
  realistically reach.
- One successful episode contributes about +311 to that episode's native
  return, so each additional success shifts `mean_return` by roughly
  311/episodes.
