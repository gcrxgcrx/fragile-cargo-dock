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
| crate_to_dock_progress | -19.626 | -0.530 | 91.406 | -16.758 | 0.757 | -15.218 | -1.241 | -1.435 | 243.830 | -1.257 | -14.049 |
| action_smoothness | 0.027 | 70.427 | 0.276 | -0.338 | -3.269 | -0.603 | -14.767 | 1.994 | 266.960 | -14.476 | 0.000 |
| total_reward | -0.004 | 0.136 | 110.437 | -0.206 | -0.519 | -4.677 | 218.489 | 248.274 | 2.086 | -7.156 | 262.218 |
| crate_docking_quality | -0.020 | -0.369 | -17.449 | 143.739 | -15.755 | 1.399 | -1.191 | -8.017 | -8.679 | 270.381 | -7.738 |
| crate_speed_penalty_near_dock | -3.828 | 50.979 | -0.460 | 125.649 | -0.832 | 201.623 | 241.319 | -14.307 | -1.129 | 249.485 | 2.244 |
| soft_contact_penalty | 24.787 | -18.634 | -1.228 | 0.524 | 187.395 | -0.790 | -7.445 | 224.539 | -1.535 | -0.129 | 241.248 |
| out_of_bounds_penalty | 48.238 | -0.051 | -0.169 | -1.312 | 167.777 | 221.511 | 1.814 | -1.970 | -13.874 | 2.121 | -1.426 |

### Reward component activation rate during training (fraction of steps where the component is non-zero)

| component | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| soft_contact_penalty | 4.5% | 100.0% | 100.0% | 100.0% | 24.0% | 41.4% | 0.3% | 0.7% | 100.0% | 100.0% | 100.0% |
| crate_docking_quality | 0.3% | 10.8% | 100.0% | 18.9% | 43.6% | 41.7% | 45.1% | 46.7% | 0.3% | 100.0% | 100.0% |
| total_reward | 2.4% | 10.9% | 0.3% | 0.2% | 100.0% | 100.0% | 23.0% | 100.0% | 47.2% | 100.0% | 0.0% |
| action_smoothness | 99.7% | 100.0% | 27.4% | 100.0% | 100.0% | 21.5% | 100.0% | 100.0% | 46.9% | 0.2% | 21.7% |
| crate_speed_penalty_near_dock | 100.0% | 4.1% | 100.0% | 100.0% | 43.3% | 100.0% | 100.0% | 26.1% | 25.4% | 49.6% | 51.8% |
| crate_to_dock_progress | 100.0% | 0.7% | 27.6% | 38.5% | 100.0% | 0.4% | 44.8% | 100.0% | 100.0% | 49.3% | 100.0% |
| out_of_bounds_penalty | 2.3% | 100.0% | 14.8% | 38.8% | 0.4% | 100.0% | 100.0% | 46.9% | 100.0% | 27.0% | 52.2% |

### Reward component values (episode sums over all training episodes)

| component | mean | abs mean | min | max |
|---|---:|---:|---:|---:|
| crate_docking_quality | -4.2193 | 161.8401 | -26.1592 | -7.5832 |
| total_reward | -0.7214 | 0.7214 | -325.0216 | 0.0000 |
| crate_to_dock_progress | 1.1148 | 1.3075 | -34.6797 | 0.0000 |
| out_of_bounds_penalty | 159.8801 | 1.1001 | -192.7733 | 393.7883 |
| soft_contact_penalty | -1.1001 | 4.2193 | 0.0000 | 366.7570 |
| action_smoothness | 180.8919 | 180.8919 | -6.6184 | 0.0000 |
| crate_speed_penalty_near_dock | -16.0857 | 16.0857 | -7.0910 | 6.8366 |