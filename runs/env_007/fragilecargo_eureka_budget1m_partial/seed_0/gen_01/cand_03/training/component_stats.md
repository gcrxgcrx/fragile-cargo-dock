# Reward Component Training Statistics

- steps_seen: 1007616
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.approach_crate | 0.002312 | 0.005317 | 0.737587 | 0.003134 | 0.007209 | -0.036582 | 0.063295 | 1007616 |
| component.bounds_guard | -0.000033 | 0.000033 | 0.001362 | -0.023996 | 0.023996 | -0.194140 | -0.000000 | 1007616 |
| component.crate_progress_toward_dock | 0.026694 | 0.029259 | 0.569249 | 0.046893 | 0.051399 | -0.173408 | 0.196571 | 1007616 |
| component.docking_settle_success | 0.369777 | 0.369777 | 0.228277 | 1.619859 | 1.619859 | 0.000000 | 3.690682 | 1007616 |
| component.fragile_handling_guard | -0.000042 | 0.000042 | 0.002161 | -0.019644 | 0.019644 | -0.298977 | -0.000000 | 1007616 |
| component.total_reward | 0.398708 | 0.402447 | 0.999968 | 0.398721 | 0.402460 | -0.267185 | 3.696086 | 1007616 |
| generated_reward | 0.398708 | 0.402447 | 0.999968 | 0.398721 | 0.402460 | -0.267185 | 3.696086 | 1007616 |
| original_env_reward | 0.019805 | 0.027103 | 1.000000 | 0.019805 | 0.027103 | -100.045974 | 300.009489 | 1007616 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_crate | 0.916784 | 0.943208 | -2.473106 | 2.375835 | 2532 |
| bounds_guard | -0.013003 | 0.013003 | -6.602425 | 0.000000 | 2532 |
| crate_progress_toward_dock | 10.596309 | 10.678739 | -4.686215 | 17.433836 | 2532 |
| docking_settle_success | 147.147269 | 147.147269 | 0.000000 | 617.312492 | 2532 |
| fragile_handling_guard | -0.016787 | 0.016787 | -0.784145 | 0.000000 | 2532 |
| total_reward | 158.630572 | 158.685907 | -6.396853 | 634.004589 | 2532 |
