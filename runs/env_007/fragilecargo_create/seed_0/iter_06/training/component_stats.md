# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.boundary_health_penalty | -0.001631 | 0.001631 | 0.006753 | -0.241569 | 0.241569 | -0.899773 | -0.000000 | 3010560 |
| component.dock_approach_improvement | 0.000021 | 0.000023 | 0.001550 | 0.013717 | 0.014818 | -0.081588 | 0.173920 | 3010560 |
| component.fragile_impact_penalty | -0.000060 | 0.000060 | 0.000106 | -0.564675 | 0.564675 | -2.173246 | -0.000000 | 3010560 |
| component.joint_dock_completion | 0.219468 | 0.219468 | 1.000000 | 0.219468 | 0.219468 | 0.097145 | 0.221042 | 3010560 |
| component.total_reward | 0.217798 | 0.219224 | 1.000000 | 0.217798 | 0.219224 | -1.919673 | 0.320249 | 3010560 |
| generated_reward | 0.217798 | 0.219224 | 1.000000 | 0.217798 | 0.219224 | -1.919673 | 0.320249 | 3010560 |
| original_env_reward | -0.005758 | 0.008287 | 1.000000 | -0.005758 | 0.008287 | -100.053546 | 0.044454 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| boundary_health_penalty | -0.651685 | 0.651685 | -179.274133 | 0.000000 | 7536 |
| dock_approach_improvement | 0.008493 | 0.009109 | -1.799200 | 10.769524 | 7536 |
| fragile_impact_penalty | -0.023903 | 0.023903 | -41.535193 | 0.000000 | 7536 |
| joint_dock_completion | 87.654198 | 87.654198 | 14.957968 | 88.416755 | 7536 |
| total_reward | 86.987103 | 87.075946 | -92.601326 | 88.416755 | 7536 |
