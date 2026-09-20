# Reward Component Training Statistics

- steps_seen: 1204224
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_smoothness_penalty | -0.009841 | 0.009841 | 1.000000 | -0.009841 | 0.009841 | -0.020000 | -0.000000 | 1204224 |
| component.crate_dock_entry_bonus | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1204224 |
| component.crate_dock_proximity | 0.075104 | 0.075104 | 1.000000 | 0.075104 | 0.075104 | 0.061861 | 0.137347 | 1204224 |
| component.crate_docking_quality | 1.029923 | 1.029923 | 1.000000 | 1.029923 | 1.029923 | 0.471891 | 1.108655 | 1204224 |
| component.crate_speed_penalty_near_dock | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | -0.000000 | 1204224 |
| component.crate_to_dock_progress | 0.000024 | 0.000025 | 0.003520 | 0.006822 | 0.006978 | -0.023028 | 0.050819 | 1204224 |
| component.obstacle_proximity_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1204224 |
| component.out_of_bounds_penalty | -0.000475 | 0.000475 | 0.015071 | -0.031515 | 0.031515 | -0.186161 | 0.000000 | 1204224 |
| component.soft_contact_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1204224 |
| component.total_reward | 1.094736 | 1.094736 | 1.000000 | 1.094736 | 1.094736 | 0.572105 | 1.198058 | 1204224 |
| generated_reward | 1.094736 | 1.094736 | 1.000000 | 1.094736 | 1.094736 | 0.572105 | 1.198058 | 1204224 |
| original_env_reward | -0.010145 | 0.013273 | 1.000000 | -0.010145 | 0.013273 | -100.047503 | 0.046447 | 1204224 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_smoothness_penalty | -3.918186 | 3.918186 | -4.464202 | -1.315707 | 3022 |
| crate_dock_entry_bonus | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3022 |
| crate_dock_proximity | 29.905119 | 29.905119 | 9.885880 | 42.101788 | 3022 |
| crate_docking_quality | 410.071271 | 410.071271 | 126.878469 | 442.545176 | 3022 |
| crate_speed_penalty_near_dock | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3022 |
| crate_to_dock_progress | 0.009569 | 0.009764 | -0.231387 | 2.908928 | 3022 |
| obstacle_proximity_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3022 |
| out_of_bounds_penalty | -0.189267 | 0.189267 | -13.758587 | 0.000000 | 3022 |
| soft_contact_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3022 |
| total_reward | 435.878506 | 435.878506 | 134.864519 | 474.580988 | 3022 |
