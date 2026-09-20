# Reward Component Training Statistics

- steps_seen: 1204224
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_smoothness | -0.019635 | 0.019635 | 1.000000 | -0.019635 | 0.019635 | -0.040000 | -0.000000 | 1204224 |
| component.crate_docking_quality | 1.797935 | 1.797935 | 0.999025 | 1.799690 | 1.799690 | 0.000000 | 5.358859 | 1204224 |
| component.crate_speed_penalty_near_dock | -0.000004 | 0.000004 | 0.005585 | -0.000720 | 0.000720 | -0.011919 | -0.000000 | 1204224 |
| component.crate_to_dock_progress | 0.002460 | 0.011881 | 0.264083 | 0.009314 | 0.044991 | -0.588870 | 0.472366 | 1204224 |
| component.out_of_bounds_penalty | -0.007766 | 0.007766 | 0.013876 | -0.559696 | 0.559696 | -3.505742 | -0.000000 | 1204224 |
| component.soft_contact_penalty | -0.000001 | 0.000001 | 0.000354 | -0.003976 | 0.003976 | -0.031570 | 0.000000 | 1204224 |
| component.total_reward | 1.772988 | 1.775372 | 1.000000 | 1.772988 | 1.775372 | -1.844814 | 5.340641 | 1204224 |
| generated_reward | 1.772988 | 1.775372 | 1.000000 | 1.772988 | 1.775372 | -1.844814 | 5.340641 | 1204224 |
| original_env_reward | -0.007141 | 0.013167 | 1.000000 | -0.007141 | 0.013167 | -100.041968 | 0.065768 | 1204224 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_smoothness | -7.823506 | 7.823506 | -12.786965 | -4.182617 | 3020 |
| crate_docking_quality | 716.441984 | 716.441984 | 0.000000 | 1645.052156 | 3020 |
| crate_speed_penalty_near_dock | -0.001602 | 0.001602 | -0.474543 | 0.000000 | 3020 |
| crate_to_dock_progress | 0.980395 | 3.790897 | -29.693713 | 40.140224 | 3020 |
| out_of_bounds_penalty | -3.096862 | 3.096862 | -302.629816 | 0.000000 | 3020 |
| soft_contact_penalty | -0.000561 | 0.000561 | -0.365185 | 0.000000 | 3020 |
| total_reward | 706.499849 | 706.736282 | -135.695444 | 1641.299880 | 3020 |
