# Reward Component Training Statistics

- steps_seen: 602112
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_smoothness_penalty | -0.019719 | 0.019719 | 1.000000 | -0.019719 | 0.019719 | -0.040000 | -0.000000 | 602112 |
| component.crate_dock_proximity | 0.250573 | 0.250573 | 1.000000 | 0.250573 | 0.250573 | 0.226582 | 0.315007 | 602112 |
| component.crate_docking_quality | 0.671975 | 0.671975 | 1.000000 | 0.671975 | 0.671975 | 0.290170 | 0.720596 | 602112 |
| component.crate_speed_penalty_near_dock | -0.000002 | 0.000002 | 0.007475 | -0.000220 | 0.000220 | -0.007022 | -0.000000 | 602112 |
| component.crate_to_dock_progress | 0.000032 | 0.000033 | 0.007608 | 0.004209 | 0.004359 | -0.013658 | 0.027285 | 602112 |
| component.obstacle_proximity_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 602112 |
| component.out_of_bounds_penalty | -0.001349 | 0.001349 | 0.034620 | -0.038963 | 0.038963 | -0.294173 | 0.000000 | 602112 |
| component.soft_contact_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 602112 |
| component.total_reward | 0.901511 | 0.901511 | 1.000000 | 0.901511 | 0.901511 | 0.524673 | 1.031153 | 602112 |
| generated_reward | 0.901511 | 0.901511 | 1.000000 | 0.901511 | 0.901511 | 0.524673 | 1.031153 | 602112 |
| original_env_reward | -0.012609 | 0.016097 | 1.000000 | -0.012609 | 0.016097 | -100.036835 | 0.043960 | 602112 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_smoothness_penalty | -7.847855 | 7.847855 | -9.119618 | -4.019549 | 1510 |
| crate_dock_proximity | 99.717516 | 99.717516 | 46.050011 | 113.119334 | 1510 |
| crate_docking_quality | 267.408714 | 267.408714 | 123.960444 | 280.318809 | 1510 |
| crate_speed_penalty_near_dock | -0.000655 | 0.000655 | -0.103005 | 0.000000 | 1510 |
| crate_to_dock_progress | 0.012770 | 0.012791 | -0.005510 | 1.880302 | 1510 |
| obstacle_proximity_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1510 |
| out_of_bounds_penalty | -0.537873 | 0.537873 | -27.037608 | 0.000000 | 1510 |
| soft_contact_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1510 |
| total_reward | 358.752615 | 358.752615 | 162.321056 | 382.322520 | 1510 |
