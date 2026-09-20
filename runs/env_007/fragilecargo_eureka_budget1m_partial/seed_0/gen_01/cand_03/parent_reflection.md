### Task score

- mean_eval_reward: 18.669944001663662
- mean_episode_length: 397.4
- eval episodes: 20
- termination breakdown: {'terminated': 1, 'truncated': 19}

### Episode return during training

| training progress | mean episode return | mean episode length |
|---|---:|---:|
| 17% | 77.75 | 396.1 |
| 33% | 74.64 | 395.9 |
| 50% | 72.06 | 395.1 |
| 67% | 69.44 | 394.8 |
| 83% | 74.72 | 393.8 |
| 100% | 68.50 | 395.8 |

### Reward component values (episode sums over all training episodes)

| component | mean | abs mean | min | max |
|---|---:|---:|---:|---:|
| bounds_guard | -0.1820 | 0.1820 | -7.5590 | 0.0000 |
| crate_progress_toward_dock | 1.8361 | 1.8724 | -4.2483 | 8.4720 |
| docking_settle_success | 71.1989 | 71.1989 | 0.0000 | 678.5494 |
| fragile_handling_guard | -0.0015 | 0.0015 | -0.5597 | 0.0000 |
| total_reward | 72.8515 | 73.0659 | -7.5051 | 686.3392 |