# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.boundary_avoidance | -0.000166 | 0.000166 | 0.008515 | -0.019459 | 0.019459 | -0.114337 | 0.000000 | 3010560 |
| component.crate_to_dock_progress | 0.021446 | 0.024505 | 0.416751 | 0.051461 | 0.058800 | -0.188288 | 0.223403 | 3010560 |
| component.joint_completion_gate | 0.050261 | 0.050261 | 0.624854 | 0.080437 | 0.080437 | 0.000000 | 0.140840 | 3010560 |
| component.soft_contact_penalty | -0.001194 | 0.001194 | 0.060279 | -0.019801 | 0.019801 | -0.077073 | 0.000000 | 3010560 |
| component.total_reward | 0.070348 | 0.071116 | 0.742130 | 0.094792 | 0.095827 | -0.172670 | 0.226315 | 3010560 |
| generated_reward | 0.070348 | 0.071116 | 0.742130 | 0.094792 | 0.095827 | -0.172670 | 0.226315 | 3010560 |
| original_env_reward | 0.010152 | 0.040065 | 1.000000 | 0.010152 | 0.040065 | -100.084499 | 300.013372 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| boundary_avoidance | -0.065386 | 0.065386 | -15.049936 | 0.000000 | 7629 |
| crate_to_dock_progress | 8.457534 | 8.463746 | -7.153342 | 14.198420 | 7629 |
| joint_completion_gate | 19.831528 | 19.831528 | 0.000000 | 36.151712 | 7629 |
| soft_contact_penalty | -0.470526 | 0.470526 | -2.810704 | 0.000000 | 7629 |
| total_reward | 27.753149 | 27.817790 | -15.049936 | 47.481719 | 7629 |
