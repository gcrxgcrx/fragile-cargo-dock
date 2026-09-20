# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.crate_to_dock_progress | 0.000035 | 0.000041 | 0.002227 | 0.015647 | 0.018503 | -0.114954 | 0.158595 | 3010560 |
| component.fragile_impact_penalty | -0.000058 | 0.000058 | 0.000124 | -0.464385 | 0.464385 | -1.511505 | -0.000000 | 3010560 |
| component.joint_dock_completion | 0.146266 | 0.146266 | 1.000000 | 0.146266 | 0.146266 | 0.031762 | 0.147361 | 3010560 |
| component.local_obstacle_penalty | -0.000153 | 0.000153 | 0.000474 | -0.322398 | 0.322398 | -0.464316 | -0.000000 | 3010560 |
| component.total_reward | 0.146091 | 0.146353 | 1.000000 | 0.146091 | 0.146353 | -1.357204 | 0.304617 | 3010560 |
| generated_reward | 0.146091 | 0.146353 | 1.000000 | 0.146091 | 0.146353 | -1.357204 | 0.304617 | 3010560 |
| original_env_reward | -0.007270 | 0.009417 | 1.000000 | -0.007270 | 0.009417 | -100.035182 | 0.041285 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| crate_to_dock_progress | 0.013908 | 0.014775 | -1.439202 | 8.483377 | 7543 |
| fragile_impact_penalty | -0.023025 | 0.023025 | -26.226788 | 0.000000 | 7543 |
| joint_dock_completion | 58.358577 | 58.358577 | 18.938702 | 58.944505 | 7543 |
| local_obstacle_penalty | -0.060949 | 0.060949 | -116.548904 | 0.000000 | 7543 |
| total_reward | 58.288511 | 58.331248 | -62.504592 | 58.944505 | 7543 |
