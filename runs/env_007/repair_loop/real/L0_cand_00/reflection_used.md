### Task score

- mean_eval_reward: -67.40267827660098
- mean_episode_length: 289.95
- eval episodes: 20
- termination breakdown: {'terminated': 13, 'truncated': 7}

### Episode return during training

| training progress | mean episode return | mean episode length |
|---|---:|---:|
| 17% | -8.11 | 166.7 |
| 33% | -8.12 | 163.7 |
| 50% | -7.97 | 166.1 |
| 67% | -7.88 | 169.4 |
| 83% | -8.20 | 169.6 |
| 100% | -7.97 | 164.7 |

### Reward component values during training (mean reward per episode)

| component | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| total_reward | -24.509 | -20.473 | -15.052 | -8.743 | -6.394 | -5.655 | -5.450 | -5.072 | -4.946 | -4.495 | -4.166 |
| out_of_bounds_penalty | -5.936 | -3.468 | -5.324 | -3.404 | -2.719 | -2.642 | -2.758 | -2.645 | -2.731 | -2.550 | -2.313 |
| action_smoothness | -18.265 | -16.570 | -9.595 | -5.318 | -3.675 | -3.013 | -2.691 | -2.427 | -2.215 | -1.944 | -1.853 |
| soft_contact_penalty | -0.289 | -0.377 | -0.136 | -0.022 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| obstacle_penalty | -0.039 | -0.081 | -0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| crate_to_dock_progress | 0.020 | 0.024 | 0.005 | 0.001 | -0.000 | -0.000 | -0.000 | -0.000 | 0.000 | 0.000 | 0.000 |
| crate_docking_quality | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| crate_speed_penalty_near_dock | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### Reward component activation rate during training (fraction of steps where the component is non-zero)

| component | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| total_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| out_of_bounds_penalty | 9.4% | 5.9% | 11.1% | 11.7% | 11.3% | 10.9% | 10.1% | 8.5% | 7.7% | 5.8% | 4.0% |
| action_smoothness | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| soft_contact_penalty | 0.6% | 0.8% | 0.3% | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| obstacle_penalty | 0.9% | 1.7% | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| crate_to_dock_progress | 4.0% | 5.8% | 1.5% | 0.4% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| crate_docking_quality | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| crate_speed_penalty_near_dock | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

### Reward component values (episode sums over all training episodes)

| component | mean | abs mean | min | max |
|---|---:|---:|---:|---:|
| action_smoothness | -4.8659 | 4.8659 | -21.8863 | -0.6822 |
| crate_docking_quality | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| crate_speed_penalty_near_dock | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| crate_to_dock_progress | 0.0024 | 0.0025 | -0.0097 | 0.7422 |
| obstacle_penalty | -0.0055 | 0.0055 | -3.9479 | 0.0000 |
| out_of_bounds_penalty | -3.1310 | 3.1310 | -97.0192 | 0.0000 |
| soft_contact_penalty | -0.0420 | 0.0420 | -27.3677 | 0.0000 |
| total_reward | -8.0420 | 8.0420 | -112.3730 | -0.6822 |