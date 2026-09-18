# FragileCargoDock-v0 baseline

- env id: `FragileCargoDock-v0`
- evaluation: 20 episodes, seeds 10000..10019
- PPO: 1,200,000 timesteps, n_envs=5, seed=0, gamma=0.99, ent_coef=0.005, arch=128,128

| policy | mean return | stdev | success | dock entered | mean steps | final goal dist (m) | final angle err (rad) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppo_native_reward | 6.18 | 1.76 | 0.0% | 0.0% | 400.0 | 2.758 | 1.053 |

## Learning curve (fixed-seed evaluation during training)

| timesteps | mean return | success | dock entered | goal dist (m) |
|---:|---:|---:|---:|---:|
| 100,000 | -9.33 | 0.0% | 0.0% | 3.917 |
| 200,000 | -20.03 | 0.0% | 0.0% | 3.729 |
| 300,000 | 1.00 | 0.0% | 0.0% | 3.735 |
| 400,000 | 1.38 | 0.0% | 0.0% | 3.486 |
| 500,000 | 3.27 | 0.0% | 0.0% | 3.572 |
| 600,000 | 4.84 | 0.0% | 0.0% | 3.422 |
| 700,000 | 5.08 | 0.0% | 0.0% | 3.229 |
| 800,000 | 5.26 | 0.0% | 0.0% | 3.462 |
| 900,000 | 4.57 | 0.0% | 0.0% | 2.809 |
| 1,000,000 | 5.53 | 0.0% | 0.0% | 2.533 |
| 1,100,000 | 6.31 | 0.0% | 10.0% | 2.437 |
| 1,200,000 | 6.10 | 0.0% | 0.0% | 2.432 |

## Convergence

- best success rate observed: 0.0%
- best mean return observed: 6.31
- first checkpoint at/above half of best success: None
- last checkpoint below half of best success: None

## Notes

- `ppo_native_reward` trains on the environment's own native reward, i.e. the
  `--use-original-reward` reference used to bound what a generated reward can
  realistically reach.
- One successful episode contributes about +311 to that episode's native
  return, so each additional success shifts `mean_return` by roughly
  311/episodes.
