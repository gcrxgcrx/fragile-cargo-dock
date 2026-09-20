### Task score

- mean_eval_reward: 1.526535374865839
- mean_episode_length: 400
- eval episodes: 20
- termination breakdown: {'terminated': 0, 'truncated': 20}

### Episode return during training

| training progress | mean episode return | mean episode length |
|---|---:|---:|
| 17% | 161.57 | 398.0 |
| 33% | 157.74 | 397.9 |
| 50% | 163.50 | 397.9 |
| 67% | 162.16 | 397.6 |
| 83% | 158.42 | 397.3 |
| 100% | 155.89 | 396.8 |

### Reward component values during training (mean reward per episode)

| component | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| crate_docking_quality | 48.238 | 70.427 | 110.437 | 143.739 | 187.395 | 221.511 | 241.319 | 248.274 | 266.960 | 270.381 | 262.218 |
| total_reward | 24.787 | 50.979 | 91.406 | 125.649 | 167.777 | 201.623 | 218.489 | 224.539 | 243.830 | 249.485 | 241.248 |
| action_smoothness | -19.626 | -18.634 | -17.449 | -16.758 | -15.755 | -15.218 | -14.767 | -14.307 | -13.874 | -14.476 | -14.049 |
| soft_contact_penalty | -0.020 | -0.369 | -1.228 | -1.312 | -3.269 | -4.677 | -7.445 | -8.017 | -8.679 | -7.156 | -7.738 |
| crate_to_dock_progress | 0.027 | 0.136 | 0.276 | 0.524 | 0.757 | 1.399 | 1.814 | 1.994 | 2.086 | 2.121 | 2.244 |
| crate_speed_penalty_near_dock | -0.004 | -0.051 | -0.169 | -0.206 | -0.519 | -0.790 | -1.241 | -1.435 | -1.535 | -1.257 | -1.426 |
| out_of_bounds_penalty | -3.828 | -0.530 | -0.460 | -0.338 | -0.832 | -0.603 | -1.191 | -1.970 | -1.129 | -0.129 | 0.000 |

### Reward component activation rate during training (fraction of steps where the component is non-zero)

| component | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| crate_docking_quality | 99.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| total_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| action_smoothness | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| soft_contact_penalty | 0.3% | 4.1% | 14.8% | 18.9% | 24.0% | 21.5% | 23.0% | 26.1% | 25.4% | 27.0% | 21.7% |
| crate_to_dock_progress | 2.4% | 10.9% | 27.6% | 38.8% | 43.6% | 41.7% | 45.1% | 46.9% | 47.2% | 49.6% | 52.2% |
| crate_speed_penalty_near_dock | 2.3% | 10.8% | 27.4% | 38.5% | 43.3% | 41.4% | 44.8% | 46.7% | 46.9% | 49.3% | 51.8% |
| out_of_bounds_penalty | 4.5% | 0.7% | 0.3% | 0.2% | 0.4% | 0.4% | 0.3% | 0.7% | 0.3% | 0.2% | 0.0% |

### Reward component values (episode sums over all training episodes)

| component | mean | abs mean | min | max |
|---|---:|---:|---:|---:|
| action_smoothness | -16.0857 | 16.0857 | -26.1592 | -7.5832 |
| crate_docking_quality | 180.8919 | 180.8919 | 0.0000 | 393.7883 |
| crate_speed_penalty_near_dock | -0.7214 | 0.7214 | -7.0910 | 0.0000 |
| crate_to_dock_progress | 1.1148 | 1.3075 | -6.6184 | 6.8366 |
| out_of_bounds_penalty | -1.1001 | 1.1001 | -325.0216 | 0.0000 |
| soft_contact_penalty | -4.2193 | 4.2193 | -34.6797 | 0.0000 |
| total_reward | 159.8801 | 161.8401 | -192.7733 | 366.7570 |