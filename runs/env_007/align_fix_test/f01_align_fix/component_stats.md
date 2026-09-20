# Reward Component Training Statistics

- steps_seen: 1204224
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_smoothness | -0.042475 | 0.042475 | 1.000000 | -0.042475 | 0.042475 | -0.100000 | -0.000000 | 1204224 |
| component.crate_docking_quality | 0.818345 | 0.818345 | 1.000000 | 0.818345 | 0.818345 | 0.004848 | 1.213971 | 1204224 |
| component.crate_speed_penalty_near_dock | -0.000005 | 0.000005 | 0.012551 | -0.000433 | 0.000433 | -0.026714 | -0.000000 | 1204224 |
| component.crate_to_dock_progress | 0.000045 | 0.000046 | 0.012790 | 0.003483 | 0.003588 | -0.015322 | 0.051054 | 1204224 |
| component.out_of_bounds_penalty | -0.002264 | 0.002264 | 0.011560 | -0.195867 | 0.195867 | -1.052806 | -0.000000 | 1204224 |
| component.soft_contact_penalty | -0.000029 | 0.000029 | 0.001649 | -0.017299 | 0.017299 | -0.174739 | 0.000000 | 1204224 |
| component.total_reward | 0.773617 | 0.773623 | 1.000000 | 0.773617 | 0.773623 | -0.291721 | 1.087558 | 1204224 |
| generated_reward | 0.773617 | 0.773623 | 1.000000 | 0.773617 | 0.773623 | -0.291721 | 1.087558 | 1204224 |
| original_env_reward | -0.008850 | 0.015082 | 1.000000 | -0.008850 | 0.015082 | -100.037250 | 0.058391 | 1204224 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_smoothness | -16.913887 | 16.913887 | -27.828145 | -7.793472 | 3022 |
| crate_docking_quality | 325.753641 | 325.753641 | 100.364314 | 376.594162 | 3022 |
| crate_speed_penalty_near_dock | -0.002167 | 0.002167 | -1.247377 | 0.000000 | 3022 |
| crate_to_dock_progress | 0.017749 | 0.018034 | -0.114092 | 3.884649 | 3022 |
| out_of_bounds_penalty | -0.902271 | 0.902271 | -65.818133 | 0.000000 | 3022 |
| soft_contact_penalty | -0.011368 | 0.011368 | -7.598285 | 0.000000 | 3022 |
| total_reward | 307.941697 | 307.941697 | 79.482021 | 362.762167 | 3022 |
