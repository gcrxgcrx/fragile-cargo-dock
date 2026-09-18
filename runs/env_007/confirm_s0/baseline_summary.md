# FragileCargoDock-v0 baseline

- env id: `FragileCargoDock-v0`
- evaluation: 100 episodes, seeds 20000..20099
- PPO: 1,500,000 timesteps, n_envs=8, seed=0, gamma=0.99, ent_coef=0.005, arch=128,128

| policy | mean return | stdev | success | dock entered | mean steps | final goal dist (m) | final angle err (rad) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppo_native_reward | 309.65 | 0.35 | 100.0% | 100.0% | 181.9 | 0.033 | 0.134 |

## Notes

- `ppo_native_reward` trains on the environment's own native reward, i.e. the
  `--use-original-reward` reference used to bound what a generated reward can
  realistically reach.
- One successful episode contributes about +311 to that episode's native
  return, so each additional success shifts `mean_return` by roughly
  311/episodes.
