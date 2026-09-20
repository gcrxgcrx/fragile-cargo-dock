# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_smoothness_penalty | -0.017424 | 0.017424 | 1.000000 | -0.017424 | 0.017424 | -0.040000 | -0.000000 | 3010560 |
| component.boundary_penalty | -0.004415 | 0.004415 | 0.143086 | -0.030854 | 0.030854 | -0.414863 | 0.000000 | 3010560 |
| component.crate_dock_alignment | -0.000086 | 0.000415 | 0.516485 | -0.000167 | 0.000804 | -0.026481 | 0.014799 | 3010560 |
| component.crate_to_dock_progress | 0.021091 | 0.024919 | 0.526347 | 0.040070 | 0.047343 | -0.167981 | 0.176299 | 3010560 |
| component.dock_proxy_reward | 0.514065 | 0.514065 | 0.528154 | 0.973325 | 0.973325 | 0.000000 | 1.381537 | 3010560 |
| component.gentle_contact_penalty | -0.019840 | 0.019840 | 0.132815 | -0.149382 | 0.149382 | -0.301670 | 0.000000 | 3010560 |
| component.obstacle_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3010560 |
| component.total_reward | 0.493391 | 0.523378 | 1.000000 | 0.493391 | 0.523378 | -0.660265 | 1.381416 | 3010560 |
| generated_reward | 0.493391 | 0.523378 | 1.000000 | 0.493391 | 0.523378 | -0.660265 | 1.381416 | 3010560 |
| original_env_reward | 0.003048 | 0.019457 | 1.000000 | 0.003048 | 0.019457 | -100.060280 | 300.011975 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_smoothness_penalty | -6.940413 | 6.940413 | -14.390982 | -4.224612 | 7554 |
| boundary_penalty | -1.759073 | 1.759073 | -53.986844 | 0.000000 | 7554 |
| crate_dock_alignment | -0.034287 | 0.037991 | -0.498747 | 0.014729 | 7554 |
| crate_to_dock_progress | 8.399583 | 8.415397 | -2.923237 | 13.169041 | 7554 |
| dock_proxy_reward | 204.799458 | 204.799458 | 0.000000 | 393.301248 | 7554 |
| gentle_contact_penalty | -7.902138 | 7.902138 | -25.615102 | 0.000000 | 7554 |
| obstacle_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 7554 |
| total_reward | 196.563130 | 201.011570 | -62.570394 | 383.347226 | 7554 |
