# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.boundary_avoidance | -0.000073 | 0.000073 | 0.004994 | -0.014709 | 0.014709 | -0.118273 | 0.000000 | 3010560 |
| component.crate_dock_alignment | 0.143426 | 0.143426 | 0.429825 | 0.333685 | 0.333685 | 0.000000 | 0.499163 | 3010560 |
| component.crate_settling | -0.016972 | 0.016972 | 0.205967 | -0.082402 | 0.082402 | -0.926858 | -0.000000 | 3010560 |
| component.crate_to_dock_progress | 0.006861 | 0.007203 | 0.618351 | 0.011096 | 0.011648 | -0.051365 | 0.058154 | 3010560 |
| component.soft_contact_penalty | -0.008709 | 0.008709 | 0.103176 | -0.084405 | 0.084405 | -0.242307 | 0.000000 | 3010560 |
| component.total_reward | 0.124533 | 0.133331 | 0.699912 | 0.177927 | 0.190496 | -0.534945 | 0.491436 | 3010560 |
| generated_reward | 0.124533 | 0.133331 | 0.699912 | 0.177927 | 0.190496 | -0.534945 | 0.491436 | 3010560 |
| original_env_reward | 0.008935 | 0.016186 | 1.000000 | 0.008935 | 0.016186 | -100.053431 | 300.014906 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| boundary_avoidance | -0.029350 | 0.029350 | -8.880222 | 0.000000 | 7535 |
| crate_dock_alignment | 57.280152 | 57.280152 | 0.000000 | 118.251066 | 7535 |
| crate_settling | -6.776966 | 6.776966 | -28.097702 | 0.000000 | 7535 |
| crate_to_dock_progress | 2.739294 | 2.762608 | -3.875872 | 4.133511 | 7535 |
| soft_contact_penalty | -3.475025 | 3.475025 | -14.939857 | 0.000000 | 7535 |
| total_reward | 49.738104 | 49.834105 | -16.763224 | 104.085869 | 7535 |
