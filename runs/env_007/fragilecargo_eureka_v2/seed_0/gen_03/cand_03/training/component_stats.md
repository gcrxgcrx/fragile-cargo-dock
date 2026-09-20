# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_smoothness_penalty | -0.005544 | 0.005544 | 1.000000 | -0.005544 | 0.005544 | -0.010000 | -0.000000 | 3010560 |
| component.boundary_penalty | -0.005648 | 0.005648 | 0.172166 | -0.032804 | 0.032804 | -0.594792 | 0.000000 | 3010560 |
| component.crate_dock_alignment | -0.010239 | 0.010239 | 0.060083 | -0.170421 | 0.170421 | -1.020000 | 0.000000 | 3010560 |
| component.crate_to_dock_progress | 0.041623 | 0.043780 | 0.436069 | 0.095451 | 0.100397 | -0.449848 | 0.352345 | 3010560 |
| component.dock_proxy_reward | 0.773425 | 0.773425 | 0.332348 | 2.327151 | 2.327151 | 0.000000 | 2.873837 | 3010560 |
| component.gentle_contact_penalty | -0.010995 | 0.010995 | 0.096604 | -0.113818 | 0.113818 | -0.295023 | 0.000000 | 3010560 |
| component.obstacle_penalty | -0.000669 | 0.000669 | 0.043100 | -0.015534 | 0.015534 | -0.067855 | 0.000000 | 3010560 |
| component.total_reward | 0.781953 | 0.818526 | 1.000000 | 0.781953 | 0.818526 | -2.254092 | 2.868204 | 3010560 |
| generated_reward | 0.781953 | 0.818526 | 1.000000 | 0.781953 | 0.818526 | -2.254092 | 2.868204 | 3010560 |
| original_env_reward | 0.002582 | 0.042347 | 1.000000 | 0.002582 | 0.042347 | -100.081353 | 300.002407 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_smoothness_penalty | -2.130271 | 2.130271 | -3.104259 | -0.288136 | 7832 |
| boundary_penalty | -2.170617 | 2.170617 | -71.467854 | 0.000000 | 7832 |
| crate_dock_alignment | -3.935672 | 3.935672 | -320.387034 | 0.000000 | 7832 |
| crate_to_dock_progress | 15.991523 | 16.063779 | -36.572529 | 27.157718 | 7832 |
| dock_proxy_reward | 297.268784 | 297.268784 | 0.000000 | 744.314749 | 7832 |
| gentle_contact_penalty | -4.223065 | 4.223065 | -17.556776 | 0.000000 | 7832 |
| obstacle_penalty | -0.257313 | 0.257313 | -19.959420 | 0.000000 | 7832 |
| total_reward | 300.543369 | 310.575519 | -352.033923 | 756.282319 | 7832 |
