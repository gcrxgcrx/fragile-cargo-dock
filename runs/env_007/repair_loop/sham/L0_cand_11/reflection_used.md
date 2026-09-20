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
| soft_contact_penalty | 262.017 | 268.794 | 0.000 | -0.569 | 268.393 | 267.943 | 99.349 | -7.194 | -7.050 | 361.016 | -0.000 |
| out_of_bounds_penalty | 0.000 | 0.000 | 360.619 | 361.068 | 0.004 | -7.210 | 360.014 | 0.000 | -0.000 | 268.432 | 0.000 |
| crate_docking_quality | -8.014 | -8.317 | -0.000 | 100.266 | -0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.000 | -6.921 |
| action_smoothness_penalty | -0.002 | -0.000 | 268.897 | 0.000 | -7.398 | -0.000 | 0.000 | 268.247 | 0.000 | -0.394 | 268.153 |
| crate_to_dock_progress | 351.293 | 360.576 | 0.000 | 268.977 | 0.000 | 359.662 | -7.253 | -0.002 | 267.962 | 99.941 | 0.012 |
| crate_dock_proximity | -1.042 | 100.280 | 0.004 | 0.006 | 0.000 | 0.001 | -0.000 | 0.018 | -1.334 | -0.000 | 99.328 |
| total_reward | 0.045 | 0.000 | -7.913 | -0.000 | 99.693 | 0.000 | 268.172 | 360.995 | -0.001 | -6.965 | 0.000 |
| obstacle_proximity_penalty | 98.289 | 0.005 | 100.061 | 0.000 | 360.227 | -0.334 | 0.000 | 100.271 | 99.494 | 0.003 | 360.563 |
| crate_speed_penalty_near_dock | 0.000 | -0.186 | -0.430 | -7.612 | -0.465 | 99.263 | -0.255 | -0.345 | 359.072 | 0.000 | -0.009 |

### Reward component activation rate during training (fraction of steps where the component is non-zero)

| component | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| total_reward | 100.0% | 0.0% | 0.0% | 4.0% | 0.0% | 100.0% | 0.0% | 3.3% | 0.0% | 100.0% | 100.0% |
| crate_docking_quality | 2.5% | 0.4% | 0.0% | 0.0% | 4.2% | 0.0% | 100.0% | 100.0% | 9.3% | 0.2% | 100.0% |
| out_of_bounds_penalty | 2.4% | 0.5% | 3.0% | 0.5% | 100.0% | 2.9% | 0.0% | 100.0% | 100.0% | 0.2% | 0.0% |
| soft_contact_penalty | 4.6% | 1.5% | 100.0% | 100.0% | 0.2% | 100.0% | 100.0% | 0.0% | 100.0% | 0.0% | 0.4% |
| obstacle_proximity_penalty | 0.0% | 100.0% | 100.0% | 0.0% | 0.2% | 0.1% | 2.6% | 1.2% | 100.0% | 100.0% | 0.0% |
| crate_dock_proximity | 100.0% | 100.0% | 100.0% | 0.5% | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| crate_to_dock_progress | 0.0% | 100.0% | 0.2% | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% | 0.1% | 0.0% | 100.0% |
| crate_speed_penalty_near_dock | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% | 1.1% | 0.1% | 100.0% | 0.0% |
| action_smoothness_penalty | 100.0% | 100.0% | 0.2% | 100.0% | 100.0% | 0.1% | 0.0% | 100.0% | 100.0% | 3.2% | 100.0% |

### Reward component values (episode sums over all training episodes)

| component | mean | abs mean | min | max |
|---|---:|---:|---:|---:|
| total_reward | 0.0085 | 359.4485 | -0.0254 | 382.8112 |
| obstacle_proximity_penalty | -0.5340 | 0.0006 | -0.1573 | 0.0000 |
| crate_docking_quality | -0.0006 | 267.7777 | 123.9604 | 1.8803 |
| crate_speed_penalty_near_dock | 99.6878 | 0.0000 | -9.1196 | -3.6128 |
| action_smoothness_penalty | -0.0000 | 99.6878 | 162.3211 | 0.0000 |
| soft_contact_penalty | 267.7777 | 0.0098 | 0.0000 | 113.1193 |
| crate_dock_proximity | -7.4909 | 0.5340 | -0.7781 | 0.0000 |
| crate_to_dock_progress | 359.4485 | 0.0000 | -27.0376 | 280.3188 |
| out_of_bounds_penalty | 0.0000 | 7.4909 | 46.0500 | 0.0000 |