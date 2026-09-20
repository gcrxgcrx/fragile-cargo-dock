# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.crate_speed_hinge | -0.000601 | 0.000601 | 0.012811 | -0.046942 | 0.046942 | -1.081105 | -0.000000 | 3010560 |
| component.dock_progress_delta | 0.065677 | 0.065818 | 0.806790 | 0.081406 | 0.081580 | -0.125030 | 0.234058 | 3010560 |
| component.fragile_impact_penalty | -0.003542 | 0.003542 | 0.089964 | -0.039375 | 0.039375 | -0.642813 | -0.000000 | 3010560 |
| component.joint_completion_gate | 0.009661 | 0.009661 | 0.800537 | 0.012068 | 0.012068 | 0.000000 | 0.034976 | 3010560 |
| component.local_obstacle_penalty | -0.001326 | 0.001326 | 0.017703 | -0.074905 | 0.074905 | -0.325529 | -0.000000 | 3010560 |
| component.total_reward | 0.069869 | 0.072104 | 0.807894 | 0.086482 | 0.089249 | -1.079910 | 0.239072 | 3010560 |
| generated_reward | 0.069869 | 0.072104 | 0.807894 | 0.086482 | 0.089249 | -1.079910 | 0.239072 | 3010560 |
| original_env_reward | 0.009157 | 0.010037 | 1.000000 | 0.009157 | 0.010037 | -100.007575 | 5.012874 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| crate_speed_hinge | -0.240595 | 0.240595 | -26.309307 | 0.000000 | 7524 |
| dock_progress_delta | 26.270426 | 26.276725 | -7.776806 | 35.203178 | 7524 |
| fragile_impact_penalty | -1.417031 | 1.417031 | -26.396598 | 0.000000 | 7524 |
| joint_completion_gate | 3.864742 | 3.864742 | 0.000000 | 5.252807 | 7524 |
| local_obstacle_penalty | -0.530595 | 0.530595 | -68.099138 | 0.000000 | 7524 |
| total_reward | 27.946947 | 28.110047 | -69.882780 | 38.343718 | 7524 |
