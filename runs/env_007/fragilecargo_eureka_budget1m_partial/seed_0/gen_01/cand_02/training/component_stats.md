# Reward Component Training Statistics

- steps_seen: 1007616
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.boundary_penalty | -0.000160 | 0.000160 | 0.006031 | -0.026498 | 0.026498 | -1.358573 | 0.000000 | 1007616 |
| component.cart_crate_approach | -0.003213 | 0.020097 | 0.998283 | -0.003218 | 0.020132 | -0.200000 | 0.200000 | 1007616 |
| component.crate_progress_toward_dock | 0.097205 | 0.103110 | 0.505082 | 0.192453 | 0.204145 | -0.670257 | 0.846949 | 1007616 |
| component.docked_settle | 0.205550 | 0.205550 | 0.105335 | 1.951397 | 1.951397 | 0.000000 | 6.122833 | 1007616 |
| component.fragile_handling_penalty | -0.000717 | 0.000717 | 0.018481 | -0.038794 | 0.038794 | -2.256852 | 0.000000 | 1007616 |
| component.settle_speed_penalty | -0.010558 | 0.010558 | 0.178285 | -0.059220 | 0.059220 | -0.698076 | -0.000000 | 1007616 |
| component.total_reward | 0.288107 | 0.313508 | 0.999894 | 0.288138 | 0.313542 | -1.957916 | 6.121519 | 1007616 |
| generated_reward | 0.288107 | 0.313508 | 0.999894 | 0.288138 | 0.313542 | -1.957916 | 6.121519 | 1007616 |
| original_env_reward | 0.008113 | 0.028856 | 1.000000 | 0.008113 | 0.028856 | -100.045156 | 299.999817 | 1007616 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| boundary_penalty | -0.063473 | 0.063473 | -24.656565 | 0.000000 | 2537 |
| cart_crate_approach | -1.276851 | 3.084256 | -12.038786 | 4.632411 | 2537 |
| crate_progress_toward_dock | 38.508850 | 38.701869 | -12.483809 | 66.335202 | 2537 |
| docked_settle | 81.381701 | 81.381701 | 0.000000 | 1172.747301 | 2537 |
| fragile_handling_penalty | -0.282355 | 0.282355 | -14.174835 | 0.000000 | 2537 |
| settle_speed_penalty | -4.182942 | 4.182942 | -22.561204 | 0.000000 | 2537 |
| total_reward | 114.084928 | 114.295822 | -20.980582 | 1217.375529 | 2537 |
