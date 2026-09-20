# Reward Component Training Statistics

- steps_seen: 1204224
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_smoothness | -0.020023 | 0.020023 | 1.000000 | -0.020023 | 0.020023 | -0.040000 | -0.000000 | 1204224 |
| component.crate_docking_quality | 1.338687 | 1.338687 | 1.000000 | 1.338687 | 1.338687 | 0.073066 | 1.506103 | 1204224 |
| component.crate_speed_penalty_near_dock | -0.000002 | 0.000002 | 0.006093 | -0.000406 | 0.000406 | -0.019023 | -0.000000 | 1204224 |
| component.crate_to_dock_progress | 0.000047 | 0.000055 | 0.006229 | 0.007585 | 0.008796 | -0.084745 | 0.080256 | 1204224 |
| component.out_of_bounds_penalty | -0.004133 | 0.004133 | 0.017342 | -0.238301 | 0.238301 | -0.873491 | -0.000000 | 1204224 |
| component.soft_contact_penalty | -0.000019 | 0.000019 | 0.000845 | -0.021898 | 0.021898 | -0.147360 | 0.000000 | 1204224 |
| component.total_reward | 1.314557 | 1.314557 | 1.000000 | 1.314557 | 1.314557 | 0.069071 | 1.489943 | 1204224 |
| generated_reward | 1.314557 | 1.314557 | 1.000000 | 1.314557 | 1.314557 | 0.069071 | 1.489943 | 1204224 |
| original_env_reward | -0.009975 | 0.013943 | 1.000000 | -0.009975 | 0.013943 | -100.034204 | 0.051431 | 1204224 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_smoothness | -7.968751 | 7.968751 | -10.936060 | -3.126852 | 3022 |
| crate_docking_quality | 532.745657 | 532.745657 | 204.132071 | 560.531664 | 3022 |
| crate_speed_penalty_near_dock | -0.000986 | 0.000986 | -1.372053 | 0.000000 | 3022 |
| crate_to_dock_progress | 0.018827 | 0.019322 | -0.517211 | 4.946335 | 3022 |
| out_of_bounds_penalty | -1.645217 | 1.645217 | -127.548669 | 0.000000 | 3022 |
| soft_contact_penalty | -0.007377 | 0.007377 | -10.519845 | 0.000000 | 3022 |
| total_reward | 523.142155 | 523.142155 | 177.619360 | 552.551730 | 3022 |
