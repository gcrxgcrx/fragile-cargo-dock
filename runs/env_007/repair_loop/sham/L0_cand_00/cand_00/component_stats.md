# Reward Component Training Statistics

- steps_seen: 1204224
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_smoothness | -0.016174 | 0.016174 | 1.000000 | -0.016174 | 0.016174 | -0.040000 | -0.000000 | 1204224 |
| component.crate_docking_quality | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1204224 |
| component.crate_to_dock_progress | 0.000071 | 0.000078 | 0.013760 | 0.005144 | 0.005670 | -0.010729 | 0.342981 | 1204224 |
| component.obstacle_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | -0.000000 | 1204224 |
| component.out_of_bounds_penalty | -0.012588 | 0.012588 | 0.077788 | -0.161823 | 0.161823 | -0.341882 | -0.000000 | 1204224 |
| component.soft_contact_penalty | -0.000017 | 0.000017 | 0.000521 | -0.032768 | 0.032768 | -0.127813 | 0.000000 | 1204224 |
| component.total_reward | -0.028708 | 0.028812 | 1.000000 | -0.028708 | 0.028812 | -0.380252 | 0.341612 | 1204224 |
| generated_reward | -0.028708 | 0.028812 | 1.000000 | -0.028708 | 0.028812 | -0.380252 | 0.341612 | 1204224 |
| original_env_reward | -0.747349 | 0.749126 | 1.000000 | -0.747349 | 0.749126 | -100.084315 | 0.063551 | 1204224 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_smoothness | -2.084259 | 2.084259 | -8.901002 | -0.638982 | 9344 |
| crate_docking_quality | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 9344 |
| crate_to_dock_progress | 0.009121 | 0.009748 | -0.603465 | 37.457659 | 9344 |
| obstacle_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 9344 |
| out_of_bounds_penalty | -1.622286 | 1.622286 | -52.142382 | 0.000000 | 9344 |
| soft_contact_penalty | -0.002202 | 0.002202 | -1.344078 | 0.000000 | 9344 |
| total_reward | -3.699626 | 3.710701 | -59.194482 | 28.302297 | 9344 |
