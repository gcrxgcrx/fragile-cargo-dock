# FragileCargoDock-v0 baseline

- env id: `FragileCargoDock-v0`
- evaluation: 20 episodes, seeds 10000..10019
- PPO: 1,500,000 timesteps, n_envs=8, seed=0, gamma=0.99, ent_coef=0.005, arch=128,128

| policy | mean return | stdev | success | dock entered | mean steps | final goal dist (m) | final angle err (rad) |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | -1.14 | 0.75 | 0.0% | 0.0% | 400.0 | 4.128 | 0.157 |
| heuristic | 158.41 | 150.25 | 50.0% | 90.0% | 373.3 | 0.107 | 0.446 |

## Notes

- `ppo_native_reward` trains on the environment's own native reward, i.e. the
  `--use-original-reward` reference used to bound what a generated reward can
  realistically reach.
- One successful episode contributes about +311 to that episode's native
  return, so each additional success shifts `mean_return` by roughly
  311/episodes.
