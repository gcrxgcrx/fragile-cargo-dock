# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_smoothness_penalty | -0.018582 | 0.018582 | 1.000000 | -0.018582 | 0.018582 | -0.040000 | -0.000000 | 3010560 |
| component.boundary_penalty | -0.004527 | 0.004527 | 0.160165 | -0.028263 | 0.028263 | -0.450952 | 0.000000 | 3010560 |
| component.crate_dock_alignment | -0.004288 | 0.004288 | 0.987253 | -0.004343 | 0.004343 | -0.599941 | -0.000000 | 3010560 |
| component.crate_to_dock_progress | 0.112275 | 0.113941 | 0.728588 | 0.154099 | 0.156386 | -0.531793 | 0.776834 | 3010560 |
| component.dock_proxy_reward | 0.155067 | 0.155067 | 0.430476 | 0.360221 | 0.360221 | 0.000000 | 0.539220 | 3010560 |
| component.gentle_contact_penalty | -0.006996 | 0.006996 | 0.105835 | -0.066099 | 0.066099 | -0.271020 | 0.000000 | 3010560 |
| component.obstacle_penalty | -0.000276 | 0.000276 | 0.009131 | -0.030182 | 0.030182 | -0.067945 | 0.000000 | 3010560 |
| component.total_reward | 0.232673 | 0.262934 | 1.000000 | 0.232673 | 0.262934 | -1.110805 | 0.664416 | 3010560 |
| generated_reward | 0.232673 | 0.262934 | 1.000000 | 0.232673 | 0.262934 | -1.110805 | 0.664416 | 3010560 |
| original_env_reward | 0.016537 | 0.025947 | 1.000000 | 0.016537 | 0.025947 | -100.070891 | 300.009482 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_smoothness_penalty | -7.398131 | 7.398131 | -12.995003 | -3.128029 | 7560 |
| boundary_penalty | -1.802179 | 1.802179 | -66.979262 | 0.000000 | 7560 |
| crate_dock_alignment | -1.707490 | 1.707490 | -154.156100 | -0.000000 | 7560 |
| crate_to_dock_progress | 44.696856 | 44.721379 | -11.000962 | 62.293631 | 7560 |
| dock_proxy_reward | 61.740630 | 61.740630 | 0.000000 | 107.329863 | 7560 |
| gentle_contact_penalty | -2.783858 | 2.783858 | -22.610609 | 0.000000 | 7560 |
| obstacle_penalty | -0.109748 | 0.109748 | -17.024912 | 0.000000 | 7560 |
| total_reward | 92.636080 | 97.073331 | -178.373332 | 145.525354 | 7560 |
