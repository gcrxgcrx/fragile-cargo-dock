# Reward Component Training Statistics

- steps_seen: 1204224
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_smoothness | -0.029164 | 0.029164 | 1.000000 | -0.029164 | 0.029164 | -0.100000 | -0.000000 | 1204224 |
| component.crate_docking_quality | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1204224 |
| component.crate_speed_penalty_near_dock | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | -0.000000 | 1204224 |
| component.crate_to_dock_progress | 0.000015 | 0.000015 | 0.011710 | 0.001251 | 0.001317 | -0.003838 | 0.013530 | 1204224 |
| component.obstacle_penalty | -0.000033 | 0.000033 | 0.002658 | -0.012505 | 0.012505 | -0.031836 | -0.000000 | 1204224 |
| component.out_of_bounds_penalty | -0.018764 | 0.018764 | 0.092069 | -0.203806 | 0.203806 | -0.440362 | -0.000000 | 1204224 |
| component.soft_contact_penalty | -0.000252 | 0.000252 | 0.001747 | -0.144226 | 0.144226 | -0.579089 | 0.000000 | 1204224 |
| component.total_reward | -0.048199 | 0.048199 | 1.000000 | -0.048199 | 0.048199 | -0.652317 | 0.006208 | 1204224 |
| generated_reward | -0.048199 | 0.048199 | 1.000000 | -0.048199 | 0.048199 | -0.652317 | 0.006208 | 1204224 |
| original_env_reward | -0.562562 | 0.565085 | 1.000000 | -0.562562 | 0.565085 | -100.081621 | 0.047800 | 1204224 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_smoothness | -4.865898 | 4.865898 | -21.886338 | -0.682237 | 7217 |
| crate_docking_quality | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 7217 |
| crate_speed_penalty_near_dock | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 7217 |
| crate_to_dock_progress | 0.002445 | 0.002456 | -0.009686 | 0.742226 | 7217 |
| obstacle_penalty | -0.005547 | 0.005547 | -3.947878 | 0.000000 | 7217 |
| out_of_bounds_penalty | -3.131001 | 3.131001 | -97.019156 | 0.000000 | 7217 |
| soft_contact_penalty | -0.042047 | 0.042047 | -27.367672 | 0.000000 | 7217 |
| total_reward | -8.042047 | 8.042047 | -112.373041 | -0.682237 | 7217 |
