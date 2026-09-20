### Task score

- mean_eval_reward: -2.258688067614172
- mean_episode_length: 400
- eval episodes: 20
- termination breakdown: {'terminated': 0, 'truncated': 20}

### Episode return during training

| training progress | mean episode return | mean episode length |
|---|---:|---:|
| 17% | 359.17 | 398.3 |
| 33% | 359.11 | 398.1 |
| 50% | 360.03 | 398.8 |
| 67% | 359.47 | 398.8 |
| 83% | 359.12 | 398.3 |
| 100% | 359.78 | 399.2 |

### Reward component values during training (mean reward per episode)

| component | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| total_reward | 351.293 | 360.576 | 360.619 | 361.068 | 360.227 | 359.662 | 360.014 | 360.995 | 359.072 | 361.016 | 360.563 |
| crate_docking_quality | 262.017 | 268.794 | 268.897 | 268.977 | 268.393 | 267.943 | 268.172 | 268.247 | 267.962 | 268.432 | 268.153 |
| crate_dock_proximity | 98.289 | 100.280 | 100.061 | 100.266 | 99.693 | 99.263 | 99.349 | 100.271 | 99.494 | 99.941 | 99.328 |
| action_smoothness_penalty | -8.014 | -8.317 | -7.913 | -7.612 | -7.398 | -7.210 | -7.253 | -7.194 | -7.050 | -6.965 | -6.921 |
| crate_to_dock_progress | 0.045 | 0.005 | 0.004 | 0.006 | 0.004 | 0.001 | -0.000 | 0.018 | -0.001 | 0.003 | 0.012 |
| out_of_bounds_penalty | -1.042 | -0.186 | -0.430 | -0.569 | -0.465 | -0.334 | -0.255 | -0.345 | -1.334 | -0.394 | -0.009 |
| crate_speed_penalty_near_dock | -0.002 | -0.000 | -0.000 | -0.000 | -0.000 | -0.000 | 0.000 | -0.002 | -0.000 | -0.000 | -0.000 |
| soft_contact_penalty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| obstacle_proximity_penalty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.000 | 0.000 |

### Reward component activation rate during training (fraction of steps where the component is non-zero)

| component | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| total_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| crate_docking_quality | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| crate_dock_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| action_smoothness_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| crate_to_dock_progress | 2.5% | 0.5% | 0.2% | 0.5% | 0.2% | 0.1% | 0.0% | 1.2% | 0.1% | 0.2% | 0.0% |
| out_of_bounds_penalty | 4.6% | 1.5% | 3.0% | 4.0% | 4.2% | 2.9% | 2.6% | 3.3% | 9.3% | 3.2% | 0.4% |
| crate_speed_penalty_near_dock | 2.4% | 0.4% | 0.2% | 0.5% | 0.2% | 0.1% | 0.0% | 1.1% | 0.1% | 0.2% | 0.0% |
| soft_contact_penalty | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| obstacle_proximity_penalty | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

### Reward component values (episode sums over all training episodes)

| component | mean | abs mean | min | max |
|---|---:|---:|---:|---:|
| action_smoothness_penalty | -7.4909 | 7.4909 | -9.1196 | -3.6128 |
| crate_dock_proximity | 99.6878 | 99.6878 | 46.0500 | 113.1193 |
| crate_docking_quality | 267.7777 | 267.7777 | 123.9604 | 280.3188 |
| crate_speed_penalty_near_dock | -0.0006 | 0.0006 | -0.1573 | 0.0000 |
| crate_to_dock_progress | 0.0085 | 0.0098 | -0.7781 | 1.8803 |
| obstacle_proximity_penalty | -0.0000 | 0.0000 | -0.0254 | 0.0000 |
| out_of_bounds_penalty | -0.5340 | 0.5340 | -27.0376 | 0.0000 |
| soft_contact_penalty | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| total_reward | 359.4485 | 359.4485 | 162.3211 | 382.8112 |