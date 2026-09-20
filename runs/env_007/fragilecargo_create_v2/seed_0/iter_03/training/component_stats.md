# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.boundary_avoidance | -0.000204 | 0.000204 | 0.010048 | -0.020311 | 0.020311 | -0.112902 | 0.000000 | 3010560 |
| component.crate_to_dock_progress | 0.010196 | 0.012957 | 0.403539 | 0.025267 | 0.032109 | -0.097197 | 0.097087 | 3010560 |
| component.joint_completion_proxy | 1.112272 | 1.112272 | 0.513718 | 2.165139 | 2.165139 | 0.000000 | 2.934198 | 3010560 |
| component.soft_contact_penalty | -0.001488 | 0.001488 | 0.055247 | -0.026940 | 0.026940 | -0.070657 | 0.000000 | 3010560 |
| component.total_reward | 1.120776 | 1.121229 | 0.674254 | 1.662245 | 1.662916 | -0.147224 | 2.934198 | 3010560 |
| generated_reward | 1.120776 | 1.121229 | 0.674254 | 1.662245 | 1.662916 | -0.147224 | 2.934198 | 3010560 |
| original_env_reward | -0.000231 | 0.028565 | 1.000000 | -0.000231 | 0.028565 | -100.080594 | 300.027893 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| boundary_avoidance | -0.081014 | 0.081014 | -6.255000 | 0.000000 | 7584 |
| crate_to_dock_progress | 4.045195 | 4.052202 | -1.234758 | 7.124243 | 7584 |
| joint_completion_proxy | 441.399114 | 441.399114 | 0.000000 | 843.023952 | 7584 |
| soft_contact_penalty | -0.590289 | 0.590289 | -4.968167 | 0.000000 | 7584 |
| total_reward | 444.773005 | 444.896110 | -6.193053 | 847.577508 | 7584 |
