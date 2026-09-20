# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.crate_settling_and_alignment | 0.401534 | 0.401534 | 0.631717 | 0.635624 | 0.635624 | 0.000000 | 0.876916 | 3010560 |
| component.crate_to_dock_progress | 0.012427 | 0.014510 | 0.421638 | 0.029473 | 0.034414 | -0.118714 | 0.120761 | 3010560 |
| component.fragile_impact_penalty | -0.000985 | 0.000985 | 0.003434 | -0.286732 | 0.286732 | -2.327965 | -0.000000 | 3010560 |
| component.out_of_bounds_penalty | -0.001611 | 0.001611 | 0.002781 | -0.579285 | 0.579285 | -2.303307 | -0.000000 | 3010560 |
| component.total_reward | 0.411366 | 0.415766 | 0.760434 | 0.540962 | 0.546748 | -2.327388 | 0.876916 | 3010560 |
| generated_reward | 0.411366 | 0.415766 | 0.760434 | 0.540962 | 0.546748 | -2.327388 | 0.876916 | 3010560 |
| original_env_reward | -0.000161 | 0.019338 | 1.000000 | -0.000161 | 0.019338 | -100.058491 | 300.001465 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| crate_settling_and_alignment | 160.184897 | 160.184897 | 0.000000 | 267.825317 | 7542 |
| crate_to_dock_progress | 4.954974 | 4.978304 | -4.895086 | 8.663298 | 7542 |
| fragile_impact_penalty | -0.393007 | 0.393007 | -61.728389 | 0.000000 | 7542 |
| out_of_bounds_penalty | -0.642958 | 0.642958 | -168.492765 | 0.000000 | 7542 |
| total_reward | 164.103906 | 165.128664 | -166.602471 | 274.993479 | 7542 |
