# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.boundary_avoidance | -0.000411 | 0.000411 | 0.019749 | -0.020825 | 0.020825 | -0.119611 | 0.000000 | 3010560 |
| component.crate_to_dock_progress | 0.012525 | 0.016062 | 0.372012 | 0.033668 | 0.043176 | -0.156132 | 0.155744 | 3010560 |
| component.joint_completion_gate | 0.310835 | 0.310835 | 0.508813 | 0.610902 | 0.610902 | 0.000000 | 0.968122 | 3010560 |
| component.soft_contact_penalty | -0.001270 | 0.001270 | 0.051850 | -0.024488 | 0.024488 | -0.073331 | 0.000000 | 3010560 |
| component.total_reward | 0.321679 | 0.322366 | 0.632601 | 0.508502 | 0.509589 | -0.139253 | 0.968122 | 3010560 |
| generated_reward | 0.321679 | 0.322366 | 0.632601 | 0.508502 | 0.509589 | -0.139253 | 0.968122 | 3010560 |
| original_env_reward | -0.009879 | 0.040098 | 1.000000 | -0.009879 | 0.040098 | -100.083033 | 300.006171 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| boundary_avoidance | -0.161218 | 0.161218 | -9.444319 | 0.000000 | 7680 |
| crate_to_dock_progress | 4.906715 | 4.913686 | -3.236448 | 10.482221 | 7680 |
| joint_completion_gate | 121.798340 | 121.798340 | 0.000000 | 259.019007 | 7680 |
| soft_contact_penalty | -0.497406 | 0.497406 | -4.033220 | 0.000000 | 7680 |
| total_reward | 126.046432 | 126.226904 | -9.444319 | 265.153151 | 7680 |
