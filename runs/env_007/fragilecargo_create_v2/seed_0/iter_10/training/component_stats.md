# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.boundary_avoidance | -0.000117 | 0.000117 | 0.007501 | -0.015593 | 0.015593 | -0.105165 | 0.000000 | 3010560 |
| component.crate_to_dock_progress | 0.021441 | 0.050120 | 0.514398 | 0.041683 | 0.097434 | -4.191436 | 3.292606 | 3010560 |
| component.joint_completion | 0.166565 | 0.177697 | 1.000000 | 0.166565 | 0.177697 | -3.851066 | 3.378049 | 3010560 |
| component.soft_contact_penalty | -0.000016 | 0.000016 | 0.001720 | -0.009406 | 0.009406 | -0.041485 | 0.000000 | 3010560 |
| component.total_reward | 0.166432 | 0.177742 | 1.000000 | 0.166432 | 0.177742 | -3.851066 | 3.371743 | 3010560 |
| generated_reward | 0.166432 | 0.177742 | 1.000000 | 0.166432 | 0.177742 | -3.851066 | 3.371743 | 3010560 |
| original_env_reward | 0.000683 | 0.018046 | 1.000000 | 0.000683 | 0.018046 | -100.070873 | 300.000720 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| boundary_avoidance | -0.046680 | 0.046680 | -11.649536 | 0.000000 | 7543 |
| crate_to_dock_progress | 8.547628 | 8.558820 | -0.194142 | 19.146578 | 7543 |
| joint_completion | 66.454306 | 66.454306 | 0.614889 | 159.376325 | 7543 |
| soft_contact_penalty | -0.006372 | 0.006372 | -0.715443 | 0.000000 | 7543 |
| total_reward | 66.401253 | 66.433861 | -9.839544 | 159.376325 | 7543 |
