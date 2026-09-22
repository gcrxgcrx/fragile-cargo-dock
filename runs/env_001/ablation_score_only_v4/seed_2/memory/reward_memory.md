# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | fuel_cost + orientation_penalty + progress | -121.77 | -121.77 | 0.00 | 68.30 | fuel_cost=-0.005 orientation_penalty=-0.013 progress=0.015 | new_best |
| 2 | crash_prevention + fuel_cost + orientation_penalty + progress | -118.52 | -118.52 | 0.00 | 68.45 | crash_prevention=-0.002 fuel_cost=-0.005 orientation_penalty=-0.013 progress=0.015 | new_best |
| 3 | crash_prevention + fuel_cost + landing_reward + orientation_penalty + progress | -122.79 | -118.52 | -4.27 | 68.30 | crash_prevention=-0.002 fuel_cost=-0.006 landing_reward=0.000 orientation_penalty=-0.013 progress=0.015 | no_meaningful_improvement |
| 4 | crash_prevention + fuel_cost + landing_reward + orientation_penalty + progress | -119.71 | -118.52 | -1.19 | 68.55 | crash_prevention=-0.002 fuel_cost=-0.006 landing_reward=0.000 orientation_penalty=-0.014 progress=0.015 | unsolved_stagnation_fresh_restart |
| 5 | contact_reward + orientation_penalty + progress + velocity_penalty | 150.41 | 150.41 | 0.00 | 873.75 | contact_reward=0.601 orientation_penalty=-0.016 progress=0.005 velocity_penalty=-0.095 | new_best |
| 6 | landing_reward + orientation_penalty + progress + velocity_penalty | -13.40 | 150.41 | -163.81 | 1000.00 | landing_reward=0.911 orientation_penalty=-0.014 progress=0.005 velocity_penalty=-0.086 | no_meaningful_improvement |
| 7 | landing_reward + orientation_penalty + progress + velocity_penalty | 13.24 | 150.41 | -137.17 | 971.25 | landing_reward=0.866 orientation_penalty=-0.010 progress=0.005 velocity_penalty=-0.008 | no_meaningful_improvement |
| 8 | landing_reward + orientation_penalty + progress + velocity_penalty | -77.47 | 150.41 | -227.88 | 908.05 | landing_reward=5.034 orientation_penalty=-0.017 progress=0.007 velocity_penalty=-0.014 | unsolved_high_achievement_continue_from_best |
| 9 | landing_reward + orientation_penalty + progress + velocity_penalty | -133.92 | 150.41 | -284.33 | 450.20 | landing_reward=4.503 orientation_penalty=-0.002 progress=0.007 velocity_penalty=-0.014 | no_meaningful_improvement |
| 10 | landing_reward + orientation_penalty + progress + velocity_penalty | -433.26 | 150.41 | -583.67 | 114.95 | landing_reward=0.007 orientation_penalty=-0.001 progress=0.006 velocity_penalty=-0.003 | no_meaningful_improvement |
