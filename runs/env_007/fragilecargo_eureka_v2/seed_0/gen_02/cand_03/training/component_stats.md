# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_smoothness_penalty | -0.018924 | 0.018924 | 1.000000 | -0.018924 | 0.018924 | -0.040000 | -0.000000 | 3010560 |
| component.boundary_penalty | -0.006624 | 0.006624 | 0.189444 | -0.034964 | 0.034964 | -0.853352 | 0.000000 | 3010560 |
| component.crate_dock_alignment | 0.195020 | 0.195020 | 0.538698 | 0.362022 | 0.362022 | 0.000000 | 0.400000 | 3010560 |
| component.crate_to_dock_progress | 0.031156 | 0.033611 | 0.493020 | 0.063194 | 0.068175 | -0.095040 | 0.246600 | 3010560 |
| component.dock_proxy_reward | 1.790050 | 1.790050 | 0.442201 | 4.048047 | 4.048047 | 0.000000 | 5.836483 | 3010560 |
| component.gentle_contact_penalty | -0.019370 | 0.019370 | 0.122532 | -0.158082 | 0.158082 | -0.309129 | 0.000000 | 3010560 |
| component.obstacle_penalty | -0.001015 | 0.001015 | 0.027154 | -0.037365 | 0.037365 | -0.116606 | 0.000000 | 3010560 |
| component.total_reward | 1.970293 | 1.999785 | 1.000000 | 1.970293 | 1.999785 | -0.944934 | 6.206296 | 3010560 |
| generated_reward | 1.970293 | 1.999785 | 1.000000 | 1.970293 | 1.999785 | -0.944934 | 6.206296 | 3010560 |
| original_env_reward | 0.003557 | 0.039807 | 1.000000 | 0.003557 | 0.039807 | -100.046558 | 300.010984 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_smoothness_penalty | -7.472377 | 7.472377 | -14.420812 | -3.540363 | 7621 |
| boundary_penalty | -2.615641 | 2.615641 | -94.208300 | 0.000000 | 7621 |
| crate_dock_alignment | 77.015636 | 77.015636 | 0.000000 | 122.309090 | 7621 |
| crate_to_dock_progress | 12.300682 | 12.336315 | -8.559262 | 18.337761 | 7621 |
| dock_proxy_reward | 706.873880 | 706.873880 | 0.000000 | 1429.181700 | 7621 |
| gentle_contact_penalty | -7.646924 | 7.646924 | -21.418303 | 0.000000 | 7621 |
| obstacle_penalty | -0.400810 | 0.400810 | -24.262738 | 0.000000 | 7621 |
| total_reward | 778.054446 | 783.149715 | -109.551675 | 1538.516380 | 7621 |
