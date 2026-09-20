# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.crate_speed_hinge | -0.000026 | 0.000026 | 0.000089 | -0.285932 | 0.285932 | -1.021489 | -0.000000 | 3010560 |
| component.crate_to_dock_progress | 0.000010 | 0.000011 | 0.002196 | 0.004779 | 0.004940 | -0.015510 | 0.049329 | 3010560 |
| component.dock_completion_joint | 10.968241 | 10.968241 | 1.000000 | 10.968241 | 10.968241 | 2.890597 | 11.052094 | 3010560 |
| component.fragile_impact_penalty | -0.000022 | 0.000022 | 0.000082 | -0.262942 | 0.262942 | -1.118648 | -0.000000 | 3010560 |
| component.total_reward | 10.968204 | 10.968204 | 1.000000 | 10.968204 | 10.968204 | 2.268224 | 11.064332 | 3010560 |
| generated_reward | 10.968204 | 10.968204 | 1.000000 | 10.968204 | 10.968204 | 2.268224 | 11.064332 | 3010560 |
| original_env_reward | -0.008501 | 0.012029 | 1.000000 | -0.008501 | 0.012029 | -100.064927 | 0.058055 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| crate_speed_hinge | -0.010175 | 0.010175 | -18.745089 | 0.000000 | 7559 |
| crate_to_dock_progress | 0.004179 | 0.004217 | -0.053706 | 2.166284 | 7559 |
| dock_completion_joint | 4367.204811 | 4367.204811 | 835.740887 | 4420.837784 | 7559 |
| fragile_impact_penalty | -0.008592 | 0.008592 | -11.544375 | 0.000000 | 7559 |
| total_reward | 4367.190223 | 4367.190223 | 835.740887 | 4420.837784 | 7559 |
