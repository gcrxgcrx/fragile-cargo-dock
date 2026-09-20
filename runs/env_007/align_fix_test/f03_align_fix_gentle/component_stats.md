# Reward Component Training Statistics

- steps_seen: 1204224
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.REPAIR_gentleness | -0.000009 | 0.000009 | 0.000716 | -0.013052 | 0.013052 | -0.078315 | -0.000000 | 1204224 |
| component.action_smoothness | -0.042853 | 0.042853 | 1.000000 | -0.042853 | 0.042853 | -0.100000 | -0.000000 | 1204224 |
| component.crate_docking_quality | 0.818632 | 0.818632 | 1.000000 | 0.818632 | 0.818632 | 0.000626 | 1.028478 | 1204224 |
| component.crate_speed_penalty_near_dock | -0.000004 | 0.000004 | 0.006145 | -0.000666 | 0.000666 | -0.023065 | -0.000000 | 1204224 |
| component.crate_to_dock_progress | 0.000029 | 0.000030 | 0.006286 | 0.004647 | 0.004831 | -0.015322 | 0.055583 | 1204224 |
| component.out_of_bounds_penalty | -0.002417 | 0.002417 | 0.010878 | -0.222205 | 0.222205 | -0.916627 | -0.000000 | 1204224 |
| component.soft_contact_penalty | -0.000026 | 0.000026 | 0.001224 | -0.020932 | 0.020932 | -0.209417 | 0.000000 | 1204224 |
| component.total_reward | 0.773351 | 0.773385 | 1.000000 | 0.773351 | 0.773385 | -0.650964 | 1.026056 | 1204224 |
| generated_reward | 0.773351 | 0.773385 | 1.000000 | 0.773351 | 0.773385 | -0.650964 | 1.026056 | 1204224 |
| original_env_reward | -0.010787 | 0.014842 | 1.000000 | -0.010787 | 0.014842 | -100.041823 | 0.047702 | 1204224 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| REPAIR_gentleness | -0.003712 | 0.003712 | -0.592704 | 0.000000 | 3031 |
| action_smoothness | -17.016323 | 17.016323 | -24.801647 | -2.646375 | 3031 |
| crate_docking_quality | 325.031716 | 325.031716 | 48.681200 | 376.594162 | 3031 |
| crate_speed_penalty_near_dock | -0.001626 | 0.001626 | -0.810275 | 0.000000 | 3031 |
| crate_to_dock_progress | 0.011607 | 0.011727 | -0.094949 | 2.977064 | 3031 |
| out_of_bounds_penalty | -0.960373 | 0.960373 | -115.624543 | 0.000000 | 3031 |
| soft_contact_penalty | -0.010180 | 0.010180 | -5.115797 | 0.000000 | 3031 |
| total_reward | 307.051109 | 307.051109 | 35.340513 | 360.534128 | 3031 |
