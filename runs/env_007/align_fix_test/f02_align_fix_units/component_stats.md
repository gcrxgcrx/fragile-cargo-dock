# Reward Component Training Statistics

- steps_seen: 1204224
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_smoothness | -0.037168 | 0.037168 | 1.000000 | -0.037168 | 0.037168 | -0.100000 | -0.000000 | 1204224 |
| component.crate_docking_quality | 0.818345 | 0.818345 | 1.000000 | 0.818345 | 0.818345 | 0.000810 | 0.954559 | 1204224 |
| component.crate_speed_penalty_near_dock | -0.000058 | 0.000058 | 0.008451 | -0.006896 | 0.006896 | -0.197192 | -0.000000 | 1204224 |
| component.crate_to_dock_progress | 0.000041 | 0.000043 | 0.008616 | 0.004746 | 0.004981 | -0.019719 | 0.050291 | 1204224 |
| component.out_of_bounds_penalty | -0.003168 | 0.003168 | 0.019344 | -0.163780 | 0.163780 | -0.913000 | -0.000000 | 1204224 |
| component.soft_contact_penalty | -0.000341 | 0.000341 | 0.001232 | -0.276747 | 0.276747 | -2.150688 | 0.000000 | 1204224 |
| component.total_reward | 0.777650 | 0.777811 | 1.000000 | 0.777650 | 0.777811 | -1.597091 | 0.954555 | 1204224 |
| generated_reward | 0.777650 | 0.777811 | 1.000000 | 0.777650 | 0.777811 | -1.597091 | 0.954555 | 1204224 |
| original_env_reward | -0.010393 | 0.014361 | 1.000000 | -0.010393 | 0.014361 | -100.043765 | 0.053115 | 1204224 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_smoothness | -14.793021 | 14.793021 | -26.164930 | -2.069213 | 3024 |
| crate_docking_quality | 325.572398 | 325.572398 | 92.185970 | 376.594162 | 3024 |
| crate_speed_penalty_near_dock | -0.023207 | 0.023207 | -5.865913 | 0.000000 | 3024 |
| crate_to_dock_progress | 0.016281 | 0.016412 | -0.085552 | 1.848059 | 3024 |
| out_of_bounds_penalty | -1.261605 | 1.261605 | -78.457808 | 0.000000 | 3024 |
| soft_contact_penalty | -0.135811 | 0.135811 | -51.504883 | 0.000000 | 3024 |
| total_reward | 309.375035 | 309.375035 | 34.097666 | 365.890505 | 3024 |
