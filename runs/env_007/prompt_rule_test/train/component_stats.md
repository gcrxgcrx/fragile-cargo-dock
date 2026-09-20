# Reward Component Training Statistics

- steps_seen: 1204224
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.crate_settle_and_align | 0.298033 | 0.298033 | 0.392561 | 0.759201 | 0.759201 | 0.000000 | 1.409104 | 1204224 |
| component.crate_to_dock_progress | 0.016037 | 0.017097 | 0.204914 | 0.078264 | 0.083434 | -0.147876 | 0.172389 | 1204224 |
| component.fragile_impact_avoidance | -0.000005 | 0.000005 | 0.000066 | -0.078673 | 0.078673 | -0.277116 | -0.000000 | 1204224 |
| component.total_reward | 0.314065 | 0.314274 | 0.539677 | 0.581949 | 0.582337 | -0.239369 | 1.409104 | 1204224 |
| generated_reward | 0.314065 | 0.314274 | 0.539677 | 0.581949 | 0.582337 | -0.239369 | 1.409104 | 1204224 |
| original_env_reward | 0.016353 | 0.034596 | 1.000000 | 0.016353 | 0.034596 | -100.049461 | 300.006504 | 1204224 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| crate_settle_and_align | 117.915315 | 117.915315 | 0.000000 | 347.576187 | 3042 |
| crate_to_dock_progress | 6.338263 | 6.341813 | -0.678065 | 11.574713 | 3042 |
| fragile_impact_avoidance | -0.002043 | 0.002043 | -1.198659 | 0.000000 | 3042 |
| total_reward | 124.251535 | 124.255488 | -0.678065 | 355.726245 | 3042 |
