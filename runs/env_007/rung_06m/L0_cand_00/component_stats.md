# Reward Component Training Statistics

- steps_seen: 602112
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_smoothness | -0.041025 | 0.041025 | 1.000000 | -0.041025 | 0.041025 | -0.100000 | -0.000000 | 602112 |
| component.crate_docking_quality | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 602112 |
| component.crate_speed_penalty_near_dock | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | -0.000000 | 602112 |
| component.crate_to_dock_progress | 0.000029 | 0.000031 | 0.023331 | 0.001257 | 0.001322 | -0.003838 | 0.013530 | 602112 |
| component.obstacle_penalty | -0.000066 | 0.000066 | 0.005316 | -0.012505 | 0.012505 | -0.031836 | -0.000000 | 602112 |
| component.out_of_bounds_penalty | -0.019270 | 0.019270 | 0.098679 | -0.195277 | 0.195277 | -0.439588 | -0.000000 | 602112 |
| component.soft_contact_penalty | -0.000504 | 0.000504 | 0.003494 | -0.144226 | 0.144226 | -0.579089 | 0.000000 | 602112 |
| component.total_reward | -0.060836 | 0.060837 | 1.000000 | -0.060836 | 0.060837 | -0.652317 | 0.006208 | 602112 |
| generated_reward | -0.060836 | 0.060837 | 1.000000 | -0.060836 | 0.060837 | -0.652317 | 0.006208 | 602112 |
| original_env_reward | -0.440314 | 0.443556 | 1.000000 | -0.440314 | 0.443556 | -100.079865 | 0.047800 | 602112 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_smoothness | -7.963837 | 7.963837 | -21.886338 | -2.346849 | 3101 |
| crate_docking_quality | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3101 |
| crate_speed_penalty_near_dock | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3101 |
| crate_to_dock_progress | 0.005693 | 0.005714 | -0.009686 | 0.742226 | 3101 |
| obstacle_penalty | -0.012909 | 0.012909 | -3.947878 | 0.000000 | 3101 |
| out_of_bounds_penalty | -3.741567 | 3.741567 | -97.019156 | 0.000000 | 3101 |
| soft_contact_penalty | -0.097856 | 0.097856 | -27.367672 | 0.000000 | 3101 |
| total_reward | -11.810476 | 11.810476 | -112.373041 | -5.239381 | 3101 |
