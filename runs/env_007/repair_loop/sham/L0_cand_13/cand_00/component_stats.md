# Reward Component Training Statistics

- steps_seen: 1204224
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_smoothness | -0.016807 | 0.016807 | 1.000000 | -0.016807 | 0.016807 | -0.060000 | -0.000000 | 1204224 |
| component.crate_docking_quality | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1204224 |
| component.crate_progress | 0.000159 | 0.000181 | 0.033588 | 0.004728 | 0.005397 | -0.028986 | 0.052868 | 1204224 |
| component.crate_proximity | 0.167307 | 0.167307 | 1.000000 | 0.167307 | 0.167307 | 0.147610 | 0.268394 | 1204224 |
| component.obstacle_proximity_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1204224 |
| component.out_of_bounds_penalty | -0.001877 | 0.001877 | 0.002024 | -0.927454 | 0.927454 | -3.246899 | 0.000000 | 1204224 |
| component.soft_contact_penalty | -0.000029 | 0.000029 | 0.000170 | -0.170210 | 0.170210 | -0.822652 | 0.000000 | 1204224 |
| component.speed_near_dock_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1204224 |
| component.total_reward | 0.148753 | 0.151990 | 1.000000 | 0.148753 | 0.151990 | -3.127937 | 0.301303 | 1204224 |
| generated_reward | 0.148753 | 0.151990 | 1.000000 | 0.148753 | 0.151990 | -3.127937 | 0.301303 | 1204224 |
| original_env_reward | -0.004202 | 0.008148 | 1.000000 | -0.004202 | 0.008148 | -100.020462 | 0.053832 | 1204224 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_smoothness | -6.722064 | 6.722064 | -13.117921 | -1.783497 | 3010 |
| crate_docking_quality | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3010 |
| crate_progress | 0.063539 | 0.067382 | -0.925795 | 4.892324 | 3010 |
| crate_proximity | 66.888658 | 66.888658 | 42.486399 | 79.585184 | 3010 |
| obstacle_proximity_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3010 |
| out_of_bounds_penalty | -0.750899 | 0.750899 | -151.527666 | 0.000000 | 3010 |
| soft_contact_penalty | -0.011592 | 0.011592 | -1.129692 | 0.000000 | 3010 |
| speed_near_dock_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3010 |
| total_reward | 59.467642 | 60.044353 | -103.055415 | 71.917619 | 3010 |
