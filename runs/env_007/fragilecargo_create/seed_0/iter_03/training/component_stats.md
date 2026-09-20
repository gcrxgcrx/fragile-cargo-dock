# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.crate_settling_and_alignment | 0.054376 | 0.054376 | 0.386241 | 0.140782 | 0.140782 | 0.000000 | 0.281395 | 3010560 |
| component.crate_to_dock_progress | 0.017646 | 0.020239 | 0.343635 | 0.051351 | 0.058895 | -0.240791 | 0.237152 | 3010560 |
| component.fragile_impact_penalty | -0.001447 | 0.001447 | 0.004209 | -0.343784 | 0.343784 | -2.368826 | -0.000000 | 3010560 |
| component.out_of_bounds_penalty | -0.001630 | 0.001630 | 0.002708 | -0.601731 | 0.601731 | -2.936429 | -0.000000 | 3010560 |
| component.total_reward | 0.068945 | 0.076201 | 0.531262 | 0.129777 | 0.143435 | -2.936429 | 0.330687 | 3010560 |
| generated_reward | 0.068945 | 0.076201 | 0.531262 | 0.129777 | 0.143435 | -2.936429 | 0.330687 | 3010560 |
| original_env_reward | 0.007135 | 0.023717 | 1.000000 | 0.007135 | 0.023717 | -100.055081 | 300.010061 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| crate_settling_and_alignment | 21.614215 | 21.614215 | 0.000000 | 75.386433 | 7572 |
| crate_to_dock_progress | 7.011631 | 7.064234 | -13.543988 | 18.199769 | 7572 |
| fragile_impact_penalty | -0.575287 | 0.575287 | -87.542716 | 0.000000 | 7572 |
| out_of_bounds_penalty | -0.647981 | 0.647981 | -227.322046 | 0.000000 | 7572 |
| total_reward | 27.402578 | 29.151832 | -227.322046 | 91.800332 | 7572 |
