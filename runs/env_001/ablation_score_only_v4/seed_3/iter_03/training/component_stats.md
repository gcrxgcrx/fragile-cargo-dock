# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.proximity_reward | 0.026657 | 0.028400 | 0.999987 | 0.026657 | 0.028401 | -0.196737 | 0.200460 | 1003520 |
| component.safe_contact_bonus | 0.018414 | 0.018414 | 0.017315 | 1.063441 | 1.063441 | 0.000000 | 1.952353 | 1003520 |
| component.soft_landing_penalty | -0.182768 | 0.182768 | 0.999998 | -0.182768 | 0.182768 | -1.585207 | -0.000000 | 1003520 |
| component.total_reward | -0.137697 | 0.165144 | 1.000000 | -0.137697 | 0.165144 | -1.608139 | 1.940452 | 1003520 |
| generated_reward | -0.137697 | 0.165144 | 1.000000 | -0.137697 | 0.165144 | -1.608139 | 1.940452 | 1003520 |
| original_env_reward | -1.596303 | 2.415006 | 1.000000 | -1.596303 | 2.415006 | -100.000000 | 141.005316 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| proximity_reward | 1.885728 | 1.885750 | -0.158388 | 2.901358 | 14185 |
| safe_contact_bonus | 1.302668 | 1.302668 | 0.000000 | 738.172828 | 14185 |
| soft_landing_penalty | -12.929269 | 12.929269 | -46.551977 | -9.403871 | 14185 |
| total_reward | -9.740872 | 9.927960 | -45.982284 | 711.202739 | 14185 |
