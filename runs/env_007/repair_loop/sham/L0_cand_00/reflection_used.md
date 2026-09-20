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
| obstacle_penalty | -5.936 | -0.377 | -9.595 | 0.000 | -2.719 | -0.000 | -2.691 | -5.072 | -2.731 | 0.000 | -1.853 |
| out_of_bounds_penalty | -0.039 | -20.473 | 0.000 | 0.000 | -3.675 | -5.655 | -0.000 | -2.645 | 0.000 | 0.000 | 0.000 |
| crate_to_dock_progress | -24.509 | -16.570 | -5.324 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -2.215 | -2.550 | 0.000 |
| action_smoothness | -0.289 | 0.000 | 0.005 | -3.404 | 0.000 | 0.000 | -5.450 | -2.427 | 0.000 | 0.000 | 0.000 |
| total_reward | -18.265 | -0.081 | 0.000 | -8.743 | -0.000 | 0.000 | -2.758 | -0.000 | 0.000 | 0.000 | 0.000 |
| soft_contact_penalty | 0.020 | -3.468 | -0.136 | 0.001 | -6.394 | -3.013 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| crate_speed_penalty_near_dock | 0.000 | 0.000 | -0.002 | -5.318 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -1.944 | -2.313 |
| crate_docking_quality | 0.000 | 0.024 | -15.052 | -0.022 | 0.000 | -2.642 | 0.000 | 0.000 | -4.946 | -4.495 | -4.166 |

### Reward component activation rate during training (fraction of steps where the component is non-zero)

| component | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| soft_contact_penalty | 100.0% | 0.8% | 0.1% | 100.0% | 0.0% | 10.9% | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% |
| action_smoothness | 0.6% | 100.0% | 0.0% | 0.0% | 11.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| crate_to_dock_progress | 0.9% | 100.0% | 100.0% | 11.7% | 0.0% | 0.0% | 10.1% | 0.0% | 0.0% | 5.8% | 4.0% |
| obstacle_penalty | 4.0% | 5.8% | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 8.5% | 7.7% | 0.0% | 100.0% |
| out_of_bounds_penalty | 9.4% | 0.0% | 11.1% | 0.4% | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| crate_docking_quality | 0.0% | 5.9% | 1.5% | 0.1% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| total_reward | 0.0% | 0.0% | 0.3% | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% | 0.0% | 100.0% | 0.0% |
| crate_speed_penalty_near_dock | 100.0% | 1.7% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 0.0% |

### Reward component values (episode sums over all training episodes)

| component | mean | abs mean | min | max |
|---|---:|---:|---:|---:|
| total_reward | 0.0000 | 0.0000 | -21.8863 | 0.7422 |
| action_smoothness | -0.0055 | 0.0055 | -27.3677 | -0.6822 |
| obstacle_penalty | -0.0420 | 0.0000 | -112.3730 | 0.0000 |
| crate_to_dock_progress | -3.1310 | 8.0420 | 0.0000 | 0.0000 |
| out_of_bounds_penalty | -4.8659 | 0.0420 | -0.0097 | 0.0000 |
| crate_speed_penalty_near_dock | -8.0420 | 0.0025 | -3.9479 | -0.6822 |
| crate_docking_quality | 0.0000 | 3.1310 | 0.0000 | 0.0000 |
| soft_contact_penalty | 0.0024 | 4.8659 | -97.0192 | 0.0000 |