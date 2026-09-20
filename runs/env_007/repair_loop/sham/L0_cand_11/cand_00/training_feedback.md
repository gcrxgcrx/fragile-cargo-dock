# Training Feedback

## Final-policy outcome
score=-2.564334, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-3.424600, -0.998343]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_docking_quality | 415.013307 | 93.2% | 93.2% | 100.0% |
| crate_dock_proximity | 30.223535 | 6.8% | 6.8% | 100.0% |
| action_smoothness_penalty | -0.226959 | -0.1% | 0.1% | 100.0% |
| crate_dock_entry_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |
| crate_speed_penalty_near_dock | 0.000000 | 0.0% | 0.0% | 0.0% |
| crate_to_dock_progress | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_proximity_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| out_of_bounds_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| soft_contact_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
