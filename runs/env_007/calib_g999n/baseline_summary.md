# FragileCargoDock-v0 baseline

- env id: `FragileCargoDock-v0`
- evaluation: 20 episodes, seeds 10000..10019
- PPO: 1,200,000 timesteps, n_envs=5, seed=0, gamma=0.999, ent_coef=0.005, arch=128,128

| policy | mean return | stdev | success | dock entered | mean steps | final goal dist (m) | final angle err (rad) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppo_native_reward | 249.96 | 121.27 | 80.0% | 90.0% | 219.1 | 0.178 | 0.254 |

## Learning curve (fixed-seed evaluation during training)

| timesteps | mean return | success | dock entered | goal dist (m) |
|---:|---:|---:|---:|---:|
| 100,000 | 0.15 | 0.0% | 0.0% | 3.883 |
| 200,000 | 0.04 | 0.0% | 0.0% | 3.698 |
| 300,000 | 0.71 | 0.0% | 0.0% | 3.695 |
| 400,000 | -26.51 | 0.0% | 0.0% | 3.212 |
| 500,000 | 5.10 | 0.0% | 0.0% | 2.231 |
| 600,000 | 7.26 | 0.0% | 20.0% | 2.399 |
| 700,000 | 7.77 | 0.0% | 30.0% | 1.848 |
| 800,000 | 9.15 | 0.0% | 80.0% | 0.849 |
| 900,000 | 8.99 | 0.0% | 80.0% | 0.808 |
| 1,000,000 | 159.41 | 50.0% | 80.0% | 0.757 |
| 1,100,000 | 189.75 | 60.0% | 90.0% | 0.139 |
| 1,200,000 | 249.90 | 80.0% | 90.0% | 0.103 |

## Convergence

- best success rate observed: 80.0%
- best mean return observed: 249.90
- first checkpoint at/above half of best success: 1000000
- last checkpoint below half of best success: 900000

## Notes

- `ppo_native_reward` trains on the environment's own native reward, i.e. the
  `--use-original-reward` reference used to bound what a generated reward can
  realistically reach.
- One successful episode contributes about +311 to that episode's native
  return, so each additional success shifts `mean_return` by roughly
  311/episodes.
