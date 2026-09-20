# Reward Component Training Statistics

- steps_seen: 602112
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_smoothness | -0.044514 | 0.044514 | 1.000000 | -0.044514 | 0.044514 | -0.100000 | -0.000000 | 602112 |
| component.crate_docking_quality | 0.283349 | 0.283349 | 0.999249 | 0.283562 | 0.283562 | 0.000000 | 1.278586 | 602112 |
| component.crate_speed_penalty_near_dock | -0.000481 | 0.000481 | 0.245840 | -0.001958 | 0.001958 | -0.075043 | -0.000000 | 602112 |
| component.crate_to_dock_progress | 0.000876 | 0.001644 | 0.247690 | 0.003538 | 0.006638 | -0.080503 | 0.066983 | 602112 |
| component.out_of_bounds_penalty | -0.003023 | 0.003023 | 0.012101 | -0.249825 | 0.249825 | -2.256742 | -0.000000 | 602112 |
| component.soft_contact_penalty | -0.003143 | 0.003143 | 0.125596 | -0.025027 | 0.025027 | -0.366277 | 0.000000 | 602112 |
| component.total_reward | 0.233064 | 0.246734 | 1.000000 | 0.233064 | 0.246734 | -1.868471 | 1.277610 | 602112 |
| generated_reward | 0.233064 | 0.246734 | 1.000000 | 0.233064 | 0.246734 | -1.868471 | 1.277610 | 602112 |
| original_env_reward | -0.009317 | 0.015570 | 1.000000 | -0.009317 | 0.015570 | -100.045240 | 5.005510 | 602112 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_smoothness | -17.642937 | 17.642937 | -23.552103 | -7.742218 | 1517 |
| crate_docking_quality | 112.217244 | 112.217244 | 0.000000 | 356.102871 | 1517 |
| crate_speed_penalty_near_dock | -0.190316 | 0.190316 | -4.560154 | 0.000000 | 1517 |
| crate_to_dock_progress | 0.345581 | 0.541574 | -6.618402 | 6.527473 | 1517 |
| out_of_bounds_penalty | -1.199884 | 1.199884 | -134.164263 | 0.000000 | 1517 |
| soft_contact_penalty | -1.242810 | 1.242810 | -25.384433 | 0.000000 | 1517 |
| total_reward | 92.286878 | 95.482549 | -101.211641 | 337.243096 | 1517 |
