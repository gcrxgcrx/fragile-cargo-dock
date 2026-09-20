# Reward Component Training Statistics

- steps_seen: 1204224
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_smoothness_penalty | -0.018792 | 0.018792 | 1.000000 | -0.018792 | 0.018792 | -0.040000 | -0.000000 | 1204224 |
| component.crate_dock_proximity | 0.250105 | 0.250105 | 1.000000 | 0.250105 | 0.250105 | 0.226476 | 0.315007 | 1204224 |
| component.crate_docking_quality | 0.671829 | 0.671829 | 1.000000 | 0.671829 | 0.671829 | 0.268416 | 0.720596 | 1204224 |
| component.crate_speed_penalty_near_dock | -0.000002 | 0.000002 | 0.005203 | -0.000293 | 0.000293 | -0.009800 | -0.000000 | 1204224 |
| component.crate_to_dock_progress | 0.000021 | 0.000025 | 0.005314 | 0.004029 | 0.004768 | -0.046918 | 0.043925 | 1204224 |
| component.obstacle_proximity_penalty | -0.000000 | 0.000000 | 0.000012 | -0.002074 | 0.002074 | -0.002909 | 0.000000 | 1204224 |
| component.out_of_bounds_penalty | -0.001338 | 0.001338 | 0.038561 | -0.034705 | 0.034705 | -0.294173 | 0.000000 | 1204224 |
| component.soft_contact_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1204224 |
| component.total_reward | 0.901823 | 0.901823 | 1.000000 | 0.901823 | 0.901823 | 0.434287 | 1.031153 | 1204224 |
| generated_reward | 0.901823 | 0.901823 | 1.000000 | 0.901823 | 0.901823 | 0.434287 | 1.031153 | 1204224 |
| original_env_reward | -0.010607 | 0.014020 | 1.000000 | -0.010607 | 0.014020 | -100.036835 | 0.044328 | 1204224 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_smoothness_penalty | -7.490887 | 7.490887 | -9.119618 | -3.612752 | 3018 |
| crate_dock_proximity | 99.687765 | 99.687765 | 46.050011 | 113.119334 | 3018 |
| crate_docking_quality | 267.777694 | 267.777694 | 123.960444 | 280.318809 | 3018 |
| crate_speed_penalty_near_dock | -0.000609 | 0.000609 | -0.157266 | 0.000000 | 3018 |
| crate_to_dock_progress | 0.008543 | 0.009842 | -0.778104 | 1.880302 | 3018 |
| obstacle_proximity_penalty | -0.000010 | 0.000010 | -0.025357 | 0.000000 | 3018 |
| out_of_bounds_penalty | -0.533990 | 0.533990 | -27.037608 | 0.000000 | 3018 |
| soft_contact_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3018 |
| total_reward | 359.448506 | 359.448506 | 162.321056 | 382.811175 | 3018 |
