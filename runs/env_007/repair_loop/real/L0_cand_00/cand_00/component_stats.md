# Reward Component Training Statistics

- steps_seen: 1204224
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_smoothness | -0.009118 | 0.009118 | 1.000000 | -0.009118 | 0.009118 | -0.020000 | -0.000000 | 1204224 |
| component.crate_approach | 0.195702 | 0.195702 | 1.000000 | 0.195702 | 0.195702 | 0.176520 | 0.246902 | 1204224 |
| component.crate_docking_quality | 1.236453 | 1.236453 | 0.999895 | 1.236583 | 1.236583 | 0.000000 | 2.207170 | 1204224 |
| component.crate_speed_penalty_near_dock | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | -0.000000 | 1204224 |
| component.crate_to_dock_progress | 0.000052 | 0.000054 | 0.002607 | 0.019831 | 0.020561 | -0.076759 | 0.157642 | 1204224 |
| component.obstacle_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | -0.000000 | 1204224 |
| component.out_of_bounds_penalty | -0.001574 | 0.001574 | 0.017624 | -0.089325 | 0.089325 | -0.313647 | -0.000000 | 1204224 |
| component.soft_contact_penalty | -0.000013 | 0.000013 | 0.000223 | -0.057107 | 0.057107 | -0.189888 | 0.000000 | 1204224 |
| component.total_reward | 1.421501 | 1.421501 | 1.000000 | 1.421501 | 1.421501 | 0.074485 | 2.405483 | 1204224 |
| generated_reward | 1.421501 | 1.421501 | 1.000000 | 1.421501 | 1.421501 | 0.074485 | 2.405483 | 1204224 |
| original_env_reward | -0.010712 | 0.014504 | 1.000000 | -0.010712 | 0.014504 | -100.044998 | 0.044554 | 1204224 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_smoothness | -3.628650 | 3.628650 | -4.480947 | -1.075015 | 3024 |
| crate_approach | 77.875027 | 77.875027 | 21.113095 | 88.551422 | 3024 |
| crate_docking_quality | 492.007937 | 492.007937 | 105.090498 | 882.868195 | 3024 |
| crate_speed_penalty_near_dock | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3024 |
| crate_to_dock_progress | 0.020591 | 0.021202 | -0.432582 | 10.202203 | 3024 |
| obstacle_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3024 |
| out_of_bounds_penalty | -0.626902 | 0.626902 | -56.381359 | 0.000000 | 3024 |
| soft_contact_penalty | -0.005061 | 0.005061 | -2.793056 | 0.000000 | 3024 |
| total_reward | 565.642943 | 565.642943 | 122.886579 | 958.767150 | 3024 |
