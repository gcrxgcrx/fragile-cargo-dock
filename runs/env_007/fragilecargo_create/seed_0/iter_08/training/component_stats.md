# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.crate_to_dock_progress | 0.060759 | 0.061124 | 0.449413 | 0.135195 | 0.136008 | -0.401549 | 0.472631 | 3010560 |
| component.fragile_impact_penalty | -0.000147 | 0.000147 | 0.000608 | -0.241710 | 0.241710 | -1.477898 | -0.000000 | 3010560 |
| component.joint_dock_completion | 0.028199 | 0.028199 | 1.000000 | 0.028199 | 0.028199 | 0.002838 | 0.049142 | 3010560 |
| component.local_obstacle_penalty | -0.000092 | 0.000092 | 0.002077 | -0.044272 | 0.044272 | -0.129671 | -0.000000 | 3010560 |
| component.total_reward | 0.088719 | 0.089157 | 1.000000 | 0.088719 | 0.089157 | -1.314851 | 0.484059 | 3010560 |
| generated_reward | 0.088719 | 0.089157 | 1.000000 | 0.088719 | 0.089157 | -1.314851 | 0.484059 | 3010560 |
| original_env_reward | 0.010428 | 0.020255 | 1.000000 | 0.010428 | 0.020255 | -100.041753 | 300.009944 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| crate_to_dock_progress | 24.214632 | 24.237870 | -4.732933 | 37.073632 | 7550 |
| fragile_impact_penalty | -0.058555 | 0.058555 | -64.722366 | 0.000000 | 7550 |
| joint_dock_completion | 11.241681 | 11.241681 | 2.460570 | 14.867377 | 7550 |
| local_obstacle_penalty | -0.036667 | 0.036667 | -6.089640 | 0.000000 | 7550 |
| total_reward | 35.361092 | 35.375609 | -40.422552 | 49.256949 | 7550 |
