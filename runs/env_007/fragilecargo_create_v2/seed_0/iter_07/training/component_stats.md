# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.boundary_avoidance | -0.000105 | 0.000105 | 0.004891 | -0.021554 | 0.021554 | -0.101467 | 0.000000 | 3010560 |
| component.crate_to_dock_progress | 0.007356 | 0.008631 | 0.440740 | 0.016690 | 0.019584 | -0.054461 | 0.058973 | 3010560 |
| component.joint_completion | 8.020828 | 8.020828 | 0.667767 | 12.011415 | 12.011415 | 0.000000 | 19.367435 | 3010560 |
| component.soft_contact_penalty | -0.001137 | 0.001137 | 0.051565 | -0.022048 | 0.022048 | -0.068394 | 0.000000 | 3010560 |
| component.total_reward | 8.026942 | 8.027073 | 0.782077 | 10.263619 | 10.263788 | -0.101467 | 19.367435 | 3010560 |
| generated_reward | 8.026942 | 8.027073 | 0.782077 | 10.263619 | 10.263788 | -0.101467 | 19.367435 | 3010560 |
| original_env_reward | 0.002041 | 0.020340 | 1.000000 | 0.002041 | 0.020340 | -100.071957 | 300.007773 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| boundary_avoidance | -0.041995 | 0.041995 | -10.331249 | 0.000000 | 7557 |
| crate_to_dock_progress | 2.928042 | 2.930760 | -1.867789 | 4.666083 | 7557 |
| joint_completion | 3194.032311 | 3194.032311 | 0.000000 | 5053.221936 | 7557 |
| soft_contact_penalty | -0.451850 | 0.451850 | -2.428233 | 0.000000 | 7557 |
| total_reward | 3196.466507 | 3196.498605 | -10.093768 | 5055.944832 | 7557 |
