# Reward Component Training Statistics

- steps_seen: 1204224
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_smoothness | -0.040456 | 0.040456 | 1.000000 | -0.040456 | 0.040456 | -0.100000 | -0.000000 | 1204224 |
| component.crate_docking_quality | 0.455072 | 0.455072 | 0.999570 | 0.455268 | 0.455268 | 0.000000 | 1.377410 | 1204224 |
| component.crate_speed_penalty_near_dock | -0.001818 | 0.001818 | 0.352018 | -0.005165 | 0.005165 | -0.194929 | -0.000000 | 1204224 |
| component.crate_to_dock_progress | 0.002813 | 0.004064 | 0.354399 | 0.007936 | 0.011468 | -0.097010 | 0.069219 | 1204224 |
| component.out_of_bounds_penalty | -0.002764 | 0.002764 | 0.007931 | -0.348545 | 0.348545 | -2.256742 | -0.000000 | 1204224 |
| component.soft_contact_penalty | -0.010635 | 0.010635 | 0.185488 | -0.057337 | 0.057337 | -0.466754 | 0.000000 | 1204224 |
| component.total_reward | 0.402211 | 0.413405 | 1.000000 | 0.402211 | 0.413405 | -2.098860 | 1.369395 | 1204224 |
| generated_reward | 0.402211 | 0.413405 | 1.000000 | 0.402211 | 0.413405 | -2.098860 | 1.369395 | 1204224 |
| original_env_reward | -0.004884 | 0.013980 | 1.000000 | -0.004884 | 0.013980 | -100.053559 | 5.018416 | 1204224 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_smoothness | -16.085747 | 16.085747 | -26.159201 | -7.583198 | 3026 |
| crate_docking_quality | 180.891890 | 180.891890 | 0.000000 | 393.788265 | 3026 |
| crate_speed_penalty_near_dock | -0.721405 | 0.721405 | -7.091022 | 0.000000 | 3026 |
| crate_to_dock_progress | 1.114777 | 1.307478 | -6.618402 | 6.836559 | 3026 |
| out_of_bounds_penalty | -1.100118 | 1.100118 | -325.021564 | 0.000000 | 3026 |
| soft_contact_penalty | -4.219326 | 4.219326 | -34.679734 | 0.000000 | 3026 |
| total_reward | 159.880072 | 161.840131 | -192.773343 | 366.756973 | 3026 |
