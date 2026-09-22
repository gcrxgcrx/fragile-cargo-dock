# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angular_penalty | -0.000967 | 0.000967 | 0.999838 | -0.000967 | 0.000967 | -4.441452 | -0.000000 | 1003520 |
| component.landing_proxy | 0.485966 | 0.485966 | 0.979995 | 0.495886 | 0.495886 | 0.000000 | 0.986261 | 1003520 |
| component.progress_reward | 0.000391 | 0.001397 | 0.999947 | 0.000391 | 0.001397 | -0.038834 | 0.040727 | 1003520 |
| component.total_reward | 0.476098 | 0.489948 | 1.000000 | 0.476098 | 0.489948 | -5.251772 | 0.986261 | 1003520 |
| component.velocity_penalty | -0.009292 | 0.009292 | 0.077270 | -0.120257 | 0.120257 | -5.228453 | -0.000000 | 1003520 |
| generated_reward | 0.476098 | 0.489948 | 1.000000 | 0.476098 | 0.489948 | -5.251772 | 0.986261 | 1003520 |
| original_env_reward | -0.228387 | 1.417887 | 1.000000 | -0.228387 | 1.417887 | -100.000000 | 104.300021 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angular_penalty | -0.646456 | 0.646456 | -9.931068 | -0.017069 | 1500 |
| landing_proxy | 324.387885 | 324.387885 | 0.107968 | 630.081819 | 1500 |
| progress_reward | 0.260850 | 0.478394 | -7.891955 | 1.417217 | 1500 |
| total_reward | 317.785733 | 322.419053 | -122.402585 | 626.355898 | 1500 |
| velocity_penalty | -6.216546 | 6.216546 | -162.918042 | 0.000000 | 1500 |
