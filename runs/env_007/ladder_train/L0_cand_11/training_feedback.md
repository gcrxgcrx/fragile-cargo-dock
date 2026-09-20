# Training Feedback

## Final-policy outcome
score=-2.258688, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-3.388405, -0.128880]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_docking_quality | 268.825448 | 72.3% | 72.3% | 100.0% |
| crate_dock_proximity | 100.610760 | 27.1% | 27.1% | 100.0% |
| action_smoothness_penalty | -1.518507 | -0.4% | 0.4% | 100.0% |
| out_of_bounds_penalty | -0.593668 | -0.2% | 0.2% | 3.4% |
| crate_to_dock_progress | 0.029595 | 0.0% | 0.0% | 1.1% |
| crate_speed_penalty_near_dock | -0.001044 | -0.0% | 0.0% | 1.1% |
| obstacle_proximity_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| soft_contact_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
