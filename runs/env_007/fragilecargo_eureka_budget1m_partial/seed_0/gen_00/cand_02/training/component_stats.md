# Reward Component Training Statistics

- steps_seen: 1007616
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.cart_bounds_guard | -0.000137 | 0.000137 | 0.000733 | -0.186464 | 0.186464 | -0.834033 | -0.000000 | 1007616 |
| component.crate_bounds_guard | -0.000120 | 0.000120 | 0.000130 | -0.923667 | 0.923667 | -2.437798 | -0.000000 | 1007616 |
| component.crate_dock_progress | 0.037500 | 0.040989 | 0.497270 | 0.075411 | 0.082427 | -0.219396 | 0.351893 | 1007616 |
| component.dock_settle_proxy | 0.330927 | 0.330927 | 0.311755 | 1.061499 | 1.061499 | 0.000000 | 1.449575 | 1007616 |
| component.fragile_impact_guard | -0.000363 | 0.000363 | 0.002820 | -0.128621 | 0.128621 | -1.534110 | -0.000000 | 1007616 |
| component.total_reward | 0.367807 | 0.369332 | 0.628719 | 0.585011 | 0.587435 | -2.812018 | 1.453793 | 1007616 |
| generated_reward | 0.367807 | 0.369332 | 0.628719 | 0.585011 | 0.587435 | -2.812018 | 1.453793 | 1007616 |
| original_env_reward | 0.012001 | 0.020790 | 1.000000 | 0.012001 | 0.020790 | -100.070116 | 300.011007 | 1007616 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| cart_bounds_guard | -0.054638 | 0.054638 | -31.389979 | 0.000000 | 2522 |
| crate_bounds_guard | -0.047978 | 0.047978 | -64.972733 | 0.000000 | 2522 |
| crate_dock_progress | 14.954865 | 14.975338 | -4.088994 | 26.626248 | 2522 |
| dock_settle_proxy | 132.038129 | 132.038129 | 0.000000 | 323.142856 | 2522 |
| fragile_impact_guard | -0.144592 | 0.144592 | -2.293076 | 0.000000 | 2522 |
| total_reward | 146.745787 | 146.938720 | -57.981602 | 342.467466 | 2522 |
