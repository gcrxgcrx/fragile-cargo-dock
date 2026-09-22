# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.proximity_reward | -0.903789 | 0.903789 | 1.000000 | -0.903789 | 0.903789 | -1.257393 | -0.000795 | 1003520 |
| component.safe_contact_bonus | 0.018579 | 0.018579 | 0.020305 | 0.914995 | 0.914995 | 0.000000 | 1.952378 | 1003520 |
| component.soft_landing_penalty | -0.294616 | 0.294616 | 0.508236 | -0.579683 | 0.579683 | -31.237604 | -0.000000 | 1003520 |
| component.total_reward | -1.179675 | 1.196132 | 1.000000 | -1.179675 | 1.196132 | -20.000000 | 1.870386 | 1003520 |
| generated_reward | -1.179675 | 1.196132 | 1.000000 | -1.179675 | 1.196132 | -20.000000 | 1.870386 | 1003520 |
| original_env_reward | -0.799174 | 3.038188 | 1.000000 | -0.799174 | 3.038188 | -100.000000 | 145.001564 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| proximity_reward | -68.832504 | 68.832504 | -758.391810 | -41.664926 | 13169 |
| safe_contact_bonus | 1.359246 | 1.359246 | 0.000000 | 639.806468 | 13169 |
| soft_landing_penalty | -22.447235 | 22.447235 | -193.620910 | -0.685730 | 13169 |
| total_reward | -89.908954 | 89.908954 | -730.308699 | -53.832869 | 13169 |
