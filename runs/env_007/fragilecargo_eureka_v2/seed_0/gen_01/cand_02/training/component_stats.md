# Reward Component Training Statistics

- steps_seen: 3010560
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_smoothness_penalty | -0.016384 | 0.016384 | 1.000000 | -0.016384 | 0.016384 | -0.040000 | -0.000000 | 3010560 |
| component.boundary_penalty | -0.003798 | 0.003798 | 0.131014 | -0.028992 | 0.028992 | -0.725277 | 0.000000 | 3010560 |
| component.crate_dock_alignment | -0.000164 | 0.000549 | 0.775626 | -0.000211 | 0.000707 | -0.047009 | 0.034004 | 3010560 |
| component.crate_to_dock_progress | 0.060958 | 0.062860 | 0.784736 | 0.077680 | 0.080103 | -0.408136 | 0.448998 | 3010560 |
| component.dock_proxy_reward | 0.252511 | 0.252511 | 0.517226 | 0.488202 | 0.488202 | 0.000000 | 0.752292 | 3010560 |
| component.gentle_contact_penalty | -0.009803 | 0.009803 | 0.104187 | -0.094093 | 0.094093 | -0.288153 | 0.000000 | 3010560 |
| component.obstacle_penalty | -0.000358 | 0.000358 | 0.014612 | -0.024520 | 0.024520 | -0.067189 | 0.000000 | 3010560 |
| component.total_reward | 0.282961 | 0.303343 | 1.000000 | 0.282961 | 0.303343 | -1.124983 | 0.742955 | 3010560 |
| generated_reward | 0.282961 | 0.303343 | 1.000000 | 0.282961 | 0.303343 | -1.124983 | 0.742955 | 3010560 |
| original_env_reward | 0.009290 | 0.017971 | 1.000000 | 0.009290 | 0.017971 | -100.053269 | 300.013321 | 3010560 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_smoothness_penalty | -6.535179 | 6.535179 | -12.924851 | -3.979049 | 7544 |
| boundary_penalty | -1.514938 | 1.514938 | -51.180829 | 0.000000 | 7544 |
| crate_dock_alignment | -0.065400 | 0.071548 | -0.999268 | 0.030074 | 7544 |
| crate_to_dock_progress | 24.306255 | 24.368546 | -35.385403 | 35.782982 | 7544 |
| dock_proxy_reward | 100.709832 | 100.709832 | 0.000000 | 174.378274 | 7544 |
| gentle_contact_penalty | -3.908091 | 3.908091 | -15.191672 | 0.000000 | 7544 |
| obstacle_penalty | -0.142980 | 0.142980 | -20.701324 | 0.000000 | 7544 |
| total_reward | 112.849498 | 115.771476 | -84.088871 | 188.649113 | 7544 |
