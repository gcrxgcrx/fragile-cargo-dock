# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.boundary_avoidance | -0.000099 | 0.000099 | 0.005134 | -0.019302 | 0.019302 | -0.059997 | 0.000000 | 3010560 |
| component.crate_to_dock_progress | 0.024278 | 0.024950 | 0.476248 | 0.050978 | 0.052388 | -0.102905 | 0.227279 | 3010560 |
| component.dock_gate | 0.423883 | 0.423883 | 0.553767 | 0.765455 | 0.765455 | 0.000000 | 0.984338 | 3010560 |
| component.soft_contact_penalty | -0.000079 | 0.000079 | 0.008069 | -0.009737 | 0.009737 | -0.069241 | 0.000000 | 3010560 |
| component.total_reward | 0.024101 | 0.024970 | 0.480390 | 0.050169 | 0.051978 | -0.128576 | 0.214018 | 3010560 |
| generated_reward | 0.024101 | 0.024970 | 0.480390 | 0.050169 | 0.051978 | -0.128576 | 0.214018 | 3010560 |
| original_env_reward | 0.032482 | 0.054002 | 1.000000 | 0.032482 | 0.054002 | -100.046679 | 300.010548 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| boundary_avoidance | -0.038884 | 0.038884 | -11.543101 | 0.000000 | 7673 |
| crate_to_dock_progress | 9.517642 | 9.534561 | -7.664559 | 14.327546 | 7673 |
| dock_gate | 166.247921 | 166.247921 | 0.000000 | 287.996765 | 7673 |
| soft_contact_penalty | -0.030824 | 0.030824 | -1.367297 | 0.000000 | 7673 |
| total_reward | 9.447933 | 9.509198 | -11.543101 | 14.327546 | 7673 |
