# Reward Component Training Statistics

- steps_seen: 602112
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_smoothness | -0.041146 | 0.041146 | 1.000000 | -0.041146 | 0.041146 | -0.100000 | -0.000000 | 602112 |
| component.crate_docking_quality | 0.429570 | 0.429570 | 1.000000 | 0.429570 | 0.429570 | 0.216762 | 0.584360 | 602112 |
| component.crate_progress | 0.000027 | 0.000028 | 0.005195 | 0.005241 | 0.005301 | -0.006833 | 0.043743 | 602112 |
| component.dock_hold_bonus | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 602112 |
| component.obstacle_proximity_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 602112 |
| component.out_of_bounds_penalty | -0.003929 | 0.003929 | 0.009199 | -0.427120 | 0.427120 | -1.499567 | 0.000000 | 602112 |
| component.soft_contact_penalty | -0.000026 | 0.000026 | 0.000281 | -0.092804 | 0.092804 | -0.519108 | 0.000000 | 602112 |
| component.speed_near_dock_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 602112 |
| component.total_reward | 0.384496 | 0.387763 | 1.000000 | 0.384496 | 0.387763 | -1.305721 | 0.584262 | 602112 |
| generated_reward | 0.384496 | 0.387763 | 1.000000 | 0.384496 | 0.387763 | -1.305721 | 0.584262 | 602112 |
| original_env_reward | -0.009507 | 0.012933 | 1.000000 | -0.009507 | 0.012933 | -100.027901 | 0.039874 | 602112 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_smoothness | -16.410449 | 16.410449 | -22.139168 | -7.487766 | 1508 |
| crate_docking_quality | 171.192128 | 171.192128 | 84.583560 | 204.671597 | 1508 |
| crate_progress | 0.010870 | 0.010968 | -0.048650 | 2.202266 | 1508 |
| dock_hold_bonus | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1508 |
| obstacle_proximity_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1508 |
| out_of_bounds_penalty | -1.568844 | 1.568844 | -90.512075 | 0.000000 | 1508 |
| soft_contact_penalty | -0.010400 | 0.010400 | -1.603640 | 0.000000 | 1508 |
| speed_near_dock_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1508 |
| total_reward | 153.213306 | 153.213306 | 0.949593 | 192.606469 | 1508 |
