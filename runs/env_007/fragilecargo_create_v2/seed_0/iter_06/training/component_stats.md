# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.boundary_avoidance | -0.000025 | 0.000025 | 0.001585 | -0.015832 | 0.015832 | -0.059983 | 0.000000 | 3010560 |
| component.completion_improvement | 0.005024 | 0.005372 | 0.309864 | 0.016212 | 0.017336 | -0.878667 | 0.182081 | 3010560 |
| component.crate_to_dock_progress | 0.020923 | 0.021345 | 0.445962 | 0.046917 | 0.047863 | -0.118246 | 0.139496 | 3010560 |
| component.soft_contact_penalty | -0.000025 | 0.000025 | 0.003132 | -0.007866 | 0.007866 | -0.043514 | 0.000000 | 3010560 |
| component.total_reward | 0.025897 | 0.026438 | 0.446986 | 0.057937 | 0.059148 | -0.862957 | 0.181806 | 3010560 |
| generated_reward | 0.025897 | 0.026438 | 0.446986 | 0.057937 | 0.059148 | -0.862957 | 0.181806 | 3010560 |
| original_env_reward | 0.009164 | 0.026369 | 1.000000 | 0.009164 | 0.026369 | -100.046646 | 300.000349 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| boundary_avoidance | -0.009987 | 0.009987 | -5.328133 | 0.000000 | 7563 |
| completion_improvement | 1.999377 | 1.999762 | -0.196863 | 3.143341 | 7563 |
| crate_to_dock_progress | 8.326633 | 8.342023 | -5.264295 | 12.022328 | 7563 |
| soft_contact_penalty | -0.009807 | 0.009807 | -1.066389 | 0.000000 | 7563 |
| total_reward | 10.306216 | 10.328566 | -5.328133 | 14.778244 | 7563 |
