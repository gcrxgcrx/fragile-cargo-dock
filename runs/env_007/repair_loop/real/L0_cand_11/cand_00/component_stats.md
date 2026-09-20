# Reward Component Training Statistics

- steps_seen: 1204224
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_smoothness_penalty | -0.020835 | 0.020835 | 1.000000 | -0.020835 | 0.020835 | -0.040000 | -0.000000 | 1204224 |
| component.crate_docking_quality | 1.342451 | 1.342451 | 1.000000 | 1.342451 | 1.342451 | 0.527736 | 1.424600 | 1204224 |
| component.crate_push_signal | 0.000002 | 0.000002 | 0.000311 | 0.004879 | 0.004879 | 0.000000 | 0.013828 | 1204224 |
| component.crate_speed_penalty_near_dock | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | -0.000000 | 1204224 |
| component.crate_to_dock_progress | 0.000068 | 0.000069 | 0.002817 | 0.024265 | 0.024525 | -0.076759 | 0.184288 | 1204224 |
| component.obstacle_proximity_penalty | -0.000000 | 0.000000 | 0.000012 | -0.002660 | 0.002660 | -0.003888 | 0.000000 | 1204224 |
| component.out_of_bounds_penalty | -0.003354 | 0.003354 | 0.044725 | -0.074999 | 0.074999 | -0.519308 | 0.000000 | 1204224 |
| component.soft_contact_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1204224 |
| component.total_reward | 1.318331 | 1.318331 | 1.000000 | 1.318331 | 1.318331 | 0.593840 | 1.422606 | 1204224 |
| generated_reward | 1.318331 | 1.318331 | 1.000000 | 1.318331 | 1.318331 | 0.593840 | 1.422606 | 1204224 |
| original_env_reward | -0.012861 | 0.014994 | 1.000000 | -0.012861 | 0.014994 | -100.037070 | 0.038169 | 1204224 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_smoothness_penalty | -8.298384 | 8.298384 | -9.463578 | -2.144296 | 3020 |
| crate_docking_quality | 534.669247 | 534.669247 | 124.711819 | 560.271740 | 3020 |
| crate_push_signal | 0.000604 | 0.000604 | 0.000000 | 0.174358 | 3020 |
| crate_speed_penalty_near_dock | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3020 |
| crate_to_dock_progress | 0.027254 | 0.027418 | -0.077765 | 8.348916 | 3020 |
| obstacle_proximity_penalty | -0.000013 | 0.000013 | -0.039905 | 0.000000 | 3020 |
| out_of_bounds_penalty | -1.337549 | 1.337549 | -56.575094 | 0.000000 | 3020 |
| soft_contact_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3020 |
| total_reward | 525.061161 | 525.061161 | 115.362595 | 551.710530 | 3020 |
