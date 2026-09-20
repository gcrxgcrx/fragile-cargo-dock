# Training Feedback

## Final-policy outcome
score=3.508158, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[2.893500, 4.010758]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_to_dock_progress | 29.470733 | 68.1% | 68.1% | 39.0% |
| joint_dock_completion | 13.723673 | 31.7% | 31.7% | 100.0% |
| fragile_impact_penalty | -0.074908 | -0.2% | 0.2% | 0.1% |
| local_obstacle_penalty | -0.020029 | -0.0% | 0.0% | 0.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
