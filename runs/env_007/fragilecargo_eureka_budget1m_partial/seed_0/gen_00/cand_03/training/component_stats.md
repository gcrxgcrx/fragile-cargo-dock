# Reward Component Training Statistics

- steps_seen: 1007616
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.crate_progress_toward_dock | 0.005438 | 0.006136 | 0.276652 | 0.019656 | 0.022179 | -0.111425 | 0.116412 | 1007616 |
| component.docking_settle_success | 0.036815 | 0.036815 | 0.065279 | 0.563961 | 0.563961 | 0.000000 | 0.785981 | 1007616 |
| component.floor_bounds_guard | -0.001412 | 0.001412 | 0.019058 | -0.074089 | 0.074089 | -0.304789 | -0.000000 | 1007616 |
| component.fragile_contact_guard | -0.000503 | 0.000503 | 0.002756 | -0.182350 | 0.182350 | -0.400000 | 0.000000 | 1007616 |
| component.push_intensity_guard | -0.001900 | 0.001900 | 0.039700 | -0.047868 | 0.047868 | -0.115802 | 0.000000 | 1007616 |
| component.steering_effort | -0.002439 | 0.002439 | 0.999979 | -0.002439 | 0.002439 | -0.010000 | -0.000000 | 1007616 |
| component.total_reward | 0.035999 | 0.044258 | 0.999985 | 0.035999 | 0.044259 | -0.463359 | 0.785981 | 1007616 |
| generated_reward | 0.035999 | 0.044258 | 0.999985 | 0.035999 | 0.044259 | -0.463359 | 0.785981 | 1007616 |
| original_env_reward | -0.043239 | 0.059493 | 1.000000 | -0.043239 | 0.059493 | -100.072593 | 299.992669 | 1007616 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| crate_progress_toward_dock | 2.004467 | 2.026361 | -4.459735 | 8.738709 | 2719 |
| docking_settle_success | 13.642284 | 13.642284 | 0.000000 | 210.658557 | 2719 |
| floor_bounds_guard | -0.523255 | 0.523255 | -28.361938 | 0.000000 | 2719 |
| fragile_contact_guard | -0.185688 | 0.185688 | -1.269973 | 0.000000 | 2719 |
| push_intensity_guard | -0.697547 | 0.697547 | -7.473584 | 0.000000 | 2719 |
| steering_effort | -0.903128 | 0.903128 | -2.273146 | -0.067593 | 2719 |
| total_reward | 13.337133 | 15.351166 | -33.184411 | 212.506814 | 2719 |
