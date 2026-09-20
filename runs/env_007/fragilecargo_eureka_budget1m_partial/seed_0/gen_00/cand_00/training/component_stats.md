# Reward Component Training Statistics

- steps_seen: 1007616
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.boundary_penalty | -0.000361 | 0.000361 | 0.005970 | -0.060491 | 0.060491 | -1.357548 | 0.000000 | 1007616 |
| component.crate_progress_toward_dock | 0.053131 | 0.057648 | 0.415480 | 0.127878 | 0.138750 | -0.472673 | 0.563277 | 1007616 |
| component.docking_settle | 0.374475 | 0.374475 | 0.192911 | 1.941183 | 1.941183 | 0.000000 | 2.776874 | 1007616 |
| component.fragile_handling_penalty | -0.001488 | 0.001488 | 0.021268 | -0.069942 | 0.069942 | -2.329646 | 0.000000 | 1007616 |
| component.total_reward | 0.425757 | 0.432545 | 0.555949 | 0.765821 | 0.778030 | -2.399777 | 2.777760 | 1007616 |
| generated_reward | 0.425757 | 0.432545 | 0.555949 | 0.765821 | 0.778030 | -2.399777 | 2.777760 | 1007616 |
| original_env_reward | 0.005522 | 0.037895 | 1.000000 | 0.005522 | 0.037895 | -100.067056 | 300.008132 | 1007616 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| boundary_penalty | -0.142019 | 0.142019 | -108.691307 | 0.000000 | 2562 |
| crate_progress_toward_dock | 20.832083 | 21.232764 | -37.037315 | 44.548850 | 2562 |
| docking_settle | 147.143006 | 147.143006 | 0.000000 | 630.233543 | 2562 |
| fragile_handling_penalty | -0.584694 | 0.584694 | -17.436204 | 0.000000 | 2562 |
| total_reward | 167.248376 | 168.137470 | -93.182000 | 666.188767 | 2562 |
