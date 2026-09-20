# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.crate_settling_and_alignment | 1.510375 | 1.510375 | 1.000000 | 1.510375 | 1.510375 | 0.191607 | 1.615168 | 3010560 |
| component.crate_to_dock_progress | 0.000009 | 0.000011 | 0.001649 | 0.005548 | 0.006672 | -0.042529 | 0.052547 | 3010560 |
| component.fragile_impact_penalty | -0.000096 | 0.000096 | 0.000134 | -0.716288 | 0.716288 | -2.412288 | -0.000000 | 3010560 |
| component.out_of_bounds_penalty | -0.004665 | 0.004665 | 0.014682 | -0.317706 | 0.317706 | -2.975478 | -0.000000 | 3010560 |
| component.total_reward | 1.505624 | 1.505711 | 1.000000 | 1.505624 | 1.505711 | -1.648435 | 1.631627 | 3010560 |
| generated_reward | 1.505624 | 1.505711 | 1.000000 | 1.505624 | 1.505711 | -1.648435 | 1.631627 | 3010560 |
| original_env_reward | -0.008148 | 0.009812 | 1.000000 | -0.008148 | 0.009812 | -100.034985 | 0.045835 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| crate_settling_and_alignment | 603.284637 | 603.284637 | 169.624621 | 629.428673 | 7536 |
| crate_to_dock_progress | 0.003654 | 0.004372 | -1.758754 | 2.815305 | 7536 |
| fragile_impact_penalty | -0.038400 | 0.038400 | -42.405069 | 0.000000 | 7536 |
| out_of_bounds_penalty | -1.863446 | 1.863446 | -219.498052 | 0.000000 | 7536 |
| total_reward | 601.386445 | 601.386445 | 111.499461 | 629.428673 | 7536 |
