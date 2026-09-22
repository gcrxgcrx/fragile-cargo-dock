# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.proximity_reward | 0.007169 | 0.007957 | 0.998549 | 0.007179 | 0.007968 | -0.154284 | 0.184395 | 1003520 |
| component.safe_contact_bonus | 0.804966 | 0.804966 | 0.468522 | 1.718096 | 1.718096 | 0.000000 | 1.998489 | 1003520 |
| component.soft_landing_penalty | -0.004103 | 0.004103 | 0.301391 | -0.013615 | 0.013615 | -0.467994 | -0.000000 | 1003520 |
| component.total_reward | 0.808031 | 0.810715 | 0.999991 | 0.808038 | 0.810722 | -0.493871 | 1.998490 | 1003520 |
| generated_reward | 0.808031 | 0.810715 | 0.999991 | 0.808038 | 0.810722 | -0.493871 | 1.998490 | 1003520 |
| original_env_reward | -0.128424 | 1.600427 | 1.000000 | -0.128424 | 1.600427 | -100.000000 | 130.961012 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| proximity_reward | 1.927411 | 1.928126 | -0.453067 | 2.928647 | 3727 |
| safe_contact_bonus | 216.014684 | 216.014684 | 0.000000 | 1729.804375 | 3727 |
| soft_landing_penalty | -1.104619 | 1.104619 | -11.472120 | 0.000000 | 3727 |
| total_reward | 216.837475 | 217.120272 | -8.005120 | 1731.726616 | 3727 |
