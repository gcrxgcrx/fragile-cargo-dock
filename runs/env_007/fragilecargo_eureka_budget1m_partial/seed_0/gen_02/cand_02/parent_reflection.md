### Task score

- mean_eval_reward: 33.7422376776659
- mean_episode_length: 379.3
- eval episodes: 20
- termination breakdown: {'terminated': 2, 'truncated': 18}

### Episode return during training

| training progress | mean episode return | mean episode length |
|---|---:|---:|
| 17% | 166.92 | 392.6 |
| 33% | 175.59 | 395.3 |
| 50% | 175.48 | 391.8 |
| 67% | 160.86 | 393.0 |
| 83% | 165.94 | 392.0 |
| 100% | 158.70 | 392.2 |

### Reward component values (episode sums over all training episodes)

| component | mean | abs mean | min | max |
|---|---:|---:|---:|---:|
| boundary_penalty | -0.1420 | 0.1420 | -108.6913 | 0.0000 |
| crate_progress_toward_dock | 20.8321 | 21.2328 | -37.0373 | 44.5488 |
| docking_settle | 147.1430 | 147.1430 | 0.0000 | 630.2335 |
| fragile_handling_penalty | -0.5847 | 0.5847 | -17.4362 | 0.0000 |
| total_reward | 167.2484 | 168.1375 | -93.1820 | 666.1888 |