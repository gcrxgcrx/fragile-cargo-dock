# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.boundary_avoidance | -0.002149 | 0.002149 | 0.095721 | -0.022450 | 0.022450 | -0.082251 | 0.000000 | 3010560 |
| component.crate_dock_alignment | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3010560 |
| component.crate_settling | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3010560 |
| component.crate_to_dock_progress | -0.208873 | 0.208873 | 1.000000 | -0.208873 | 0.208873 | -0.259378 | -0.126870 | 3010560 |
| component.soft_contact_penalty | -0.000000 | 0.000000 | 0.000003 | -0.006198 | 0.006198 | -0.009262 | 0.000000 | 3010560 |
| component.total_reward | -0.211021 | 0.211021 | 1.000000 | -0.211021 | 0.211021 | -0.299504 | -0.126870 | 3010560 |
| generated_reward | -0.211021 | 0.211021 | 1.000000 | -0.211021 | 0.211021 | -0.299504 | -0.126870 | 3010560 |
| original_env_reward | -1.305932 | 1.313328 | 1.000000 | -1.305932 | 1.313328 | -100.088075 | 0.055876 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| boundary_avoidance | -0.168505 | 0.168505 | -6.979346 | 0.000000 | 38393 |
| crate_dock_alignment | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 38393 |
| crate_settling | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 38393 |
| crate_to_dock_progress | -16.376649 | 16.376649 | -94.895467 | -10.774574 | 38393 |
| soft_contact_penalty | -0.000001 | 0.000001 | -0.049583 | 0.000000 | 38393 |
| total_reward | -16.545155 | 16.545155 | -98.179124 | -10.927710 | 38393 |
