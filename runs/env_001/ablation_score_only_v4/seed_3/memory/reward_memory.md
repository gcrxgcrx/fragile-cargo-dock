# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | proximity_reward + safe_contact_bonus + soft_landing_penalty | 5.67 | 5.67 | 0.00 | 84.40 | proximity_reward=-0.904 safe_contact_bonus=0.019 soft_landing_penalty=-0.295 | new_best |
| 2 | proximity_reward + safe_contact_bonus + soft_landing_penalty | -119.47 | 5.67 | -125.14 | 68.35 | proximity_reward=-0.934 safe_contact_bonus=0.017 soft_landing_penalty=-0.185 | no_meaningful_improvement |
| 3 | proximity_reward + safe_contact_bonus + soft_landing_penalty | -111.71 | 5.67 | -117.37 | 68.45 | proximity_reward=0.027 safe_contact_bonus=0.018 soft_landing_penalty=-0.183 | no_meaningful_improvement |
| 4 | proximity_reward + safe_contact_bonus + soft_landing_penalty | -9.34 | 5.67 | -15.00 | 1000.00 | proximity_reward=0.007 safe_contact_bonus=0.805 soft_landing_penalty=-0.004 | same_skeleton_oscillation_fresh_restart |
| 5 | distance_shaping + fuel_penalty + orientation_penalty + progress_reward + velocity_constraint | -74.79 | 5.67 | -80.46 | 1000.00 | distance_shaping=1.668 fuel_penalty=-0.055 orientation_penalty=-0.026 progress_reward=0.019 velocity_constraint=-0.002 | no_meaningful_improvement |
| 6 | distance_shaping + fuel_penalty + landing_bonus + orientation_penalty + progress_reward + velocity_constraint | 50.66 | 50.66 | 0.00 | 703.50 | distance_shaping=1.614 fuel_penalty=-0.055 landing_bonus=1.068 orientation_penalty=-0.026 progress_reward=0.018 | new_best |
| 7 | distance_shaping + fuel_penalty + landing_quality + orientation_penalty + progress_reward + velocity_constraint | -37.65 | 50.66 | -88.31 | 1000.00 | distance_shaping=1.649 fuel_penalty=-0.049 landing_quality=0.121 orientation_penalty=-0.018 progress_reward=0.020 | no_meaningful_improvement |
| 8 | approach_and_land + distance_shaping + efficiency_penalty + orientation_penalty + progress_reward + velocity_constraint | -39.40 | 50.66 | -90.06 | 1000.00 | approach_and_land=0.663 distance_shaping=1.618 efficiency_penalty=-0.025 orientation_penalty=-0.025 progress_reward=0.020 | no_meaningful_improvement |
| 9 | distance_shaping + efficiency_penalty + orientation_penalty + progress_reward + safe_approach + velocity_constraint | -12.71 | 50.66 | -63.37 | 964.85 | distance_shaping=1.582 efficiency_penalty=-0.028 orientation_penalty=-0.019 progress_reward=0.018 safe_approach=1.314 | unsolved_high_achievement_continue_from_best |
| 10 | distance_shaping + efficiency_penalty + landing_bonus + orientation_penalty + progress_reward + velocity_constraint | -72.94 | 50.66 | -123.60 | 1000.00 | distance_shaping=1.605 efficiency_penalty=-0.020 landing_bonus=0.591 orientation_penalty=-0.022 progress_reward=0.021 | no_meaningful_improvement |
