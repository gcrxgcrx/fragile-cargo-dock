# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | fuel_penalty + goal_proximity + safe_landing_penalty | -11.79 | -11.79 | 0.00 | 1000.00 | fuel_penalty=-0.008 goal_proximity=1.638 safe_landing_penalty=-0.071 | new_best |
| 2 | fuel_penalty + landing_proxy + safe_landing_penalty | -37.08 | -11.79 | -25.28 | 1000.00 | fuel_penalty=-0.008 landing_proxy=1.851 safe_landing_penalty=-0.017 | no_meaningful_improvement |
| 3 | landing_proxy + safe_landing_penalty | -9.21 | -9.21 | 0.00 | 1000.00 | landing_proxy=1.224 safe_landing_penalty=-0.001 | new_best |
| 4 | potential_gain + safe_landing_penalty | 261.39 | 261.39 | 0.00 | 325.70 | potential_gain=0.006 safe_landing_penalty=-0.003 | target_solved_new_best |
| 5 | action_efficiency_cost + potential_gain + safe_landing_penalty | -115.81 | 261.39 | -377.20 | 68.40 | action_efficiency_cost=-0.001 potential_gain=0.013 safe_landing_penalty=-0.010 | stop_after_solved_drop_keep_best |
