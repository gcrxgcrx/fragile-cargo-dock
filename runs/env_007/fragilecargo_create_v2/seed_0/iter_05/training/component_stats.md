# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.boundary_avoidance | -0.000154 | 0.000154 | 0.006390 | -0.024054 | 0.024054 | -0.117565 | 0.000000 | 3010560 |
| component.crate_to_dock_progress | 0.017946 | 0.021502 | 0.426447 | 0.042082 | 0.050421 | -0.132143 | 0.146736 | 3010560 |
| component.joint_completion | 1.030411 | 1.030411 | 0.589209 | 1.748805 | 1.748805 | 0.000000 | 2.855480 | 3010560 |
| component.soft_contact_penalty | -0.001183 | 0.001183 | 0.055473 | -0.021322 | 0.021322 | -0.067282 | 0.000000 | 3010560 |
| component.total_reward | 1.047020 | 1.047473 | 0.740840 | 1.413288 | 1.413899 | -0.133663 | 2.857135 | 3010560 |
| generated_reward | 1.047020 | 1.047473 | 0.740840 | 1.413288 | 1.413899 | -0.133663 | 2.857135 | 3010560 |
| original_env_reward | 0.009676 | 0.036021 | 1.000000 | 0.009676 | 0.036021 | -100.589136 | 300.010527 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| boundary_avoidance | -0.060857 | 0.060857 | -11.942919 | 0.000000 | 7603 |
| crate_to_dock_progress | 7.099862 | 7.114956 | -4.662251 | 10.899162 | 7603 |
| joint_completion | 407.786879 | 407.786879 | 0.000000 | 760.992566 | 7603 |
| soft_contact_penalty | -0.467648 | 0.467648 | -2.404380 | 0.000000 | 7603 |
| total_reward | 414.358235 | 414.418453 | -12.024813 | 769.080779 | 7603 |
