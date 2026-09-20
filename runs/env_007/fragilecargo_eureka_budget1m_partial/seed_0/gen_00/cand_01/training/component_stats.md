# Reward Component Training Statistics

- steps_seen: 1007616
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.bounds_guard | -0.000460 | 0.000460 | 0.013232 | -0.034787 | 0.034787 | -0.111735 | -0.000000 | 1007616 |
| component.crate_progress_toward_dock | 0.004651 | 0.006359 | 0.358520 | 0.012974 | 0.017738 | -0.089447 | 0.099176 | 1007616 |
| component.docking_settle_success | 0.180044 | 0.180044 | 0.091689 | 1.963641 | 1.963641 | 0.000000 | 2.922330 | 1007616 |
| component.fragile_handling_guard | -0.000004 | 0.000004 | 0.000198 | -0.018651 | 0.018651 | -0.297041 | -0.000000 | 1007616 |
| component.total_reward | 0.184231 | 0.185996 | 0.379792 | 0.485085 | 0.489733 | -0.285910 | 2.921419 | 1007616 |
| generated_reward | 0.184231 | 0.185996 | 0.379792 | 0.485085 | 0.489733 | -0.285910 | 2.921419 | 1007616 |
| original_env_reward | -0.011871 | 0.032191 | 1.000000 | -0.011871 | 0.032191 | -100.049976 | 300.002429 | 1007616 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| bounds_guard | -0.182031 | 0.182031 | -7.558981 | 0.000000 | 2548 |
| crate_progress_toward_dock | 1.836109 | 1.872442 | -4.248253 | 8.471953 | 2548 |
| docking_settle_success | 71.198924 | 71.198924 | 0.000000 | 678.549395 | 2548 |
| fragile_handling_guard | -0.001464 | 0.001464 | -0.559695 | 0.000000 | 2548 |
| total_reward | 72.851538 | 73.065948 | -7.505133 | 686.339208 | 2548 |
