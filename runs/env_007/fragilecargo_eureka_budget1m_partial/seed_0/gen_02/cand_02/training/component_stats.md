# Reward Component Training Statistics

- steps_seen: 1007616
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.boundary_penalty | -0.000376 | 0.000376 | 0.002264 | -0.166195 | 0.166195 | -1.360394 | 0.000000 | 1007616 |
| component.crate_progress_toward_dock | 0.108611 | 0.115990 | 0.489605 | 0.221834 | 0.236905 | -0.654123 | 0.857243 | 1007616 |
| component.docking_completion | 0.030045 | 0.030045 | 0.030578 | 0.982564 | 0.982564 | 0.000000 | 3.052901 | 1007616 |
| component.docking_settle | 0.173106 | 0.173106 | 0.192525 | 0.899134 | 0.899134 | 0.000000 | 1.289935 | 1007616 |
| component.fragile_handling_penalty | -0.000817 | 0.000817 | 0.012163 | -0.067196 | 0.067196 | -2.764186 | 0.000000 | 1007616 |
| component.total_reward | 0.310568 | 0.318926 | 0.621514 | 0.499696 | 0.513144 | -2.471497 | 4.352294 | 1007616 |
| generated_reward | 0.310568 | 0.318926 | 0.621514 | 0.499696 | 0.513144 | -2.471497 | 4.352294 | 1007616 |
| original_env_reward | 0.034338 | 0.045004 | 1.000000 | 0.034338 | 0.045004 | -100.047307 | 300.008884 | 1007616 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| boundary_penalty | -0.148430 | 0.148430 | -50.236906 | 0.000000 | 2554 |
| crate_progress_toward_dock | 42.785986 | 42.855657 | -11.277816 | 67.827828 | 2554 |
| docking_completion | 11.813835 | 11.813835 | 0.000000 | 447.252220 | 2554 |
| docking_settle | 68.143592 | 68.143592 | 0.000000 | 275.059626 | 2554 |
| fragile_handling_penalty | -0.322456 | 0.322456 | -12.717416 | 0.000000 | 2554 |
| total_reward | 122.272527 | 122.437488 | -27.567183 | 745.538355 | 2554 |
