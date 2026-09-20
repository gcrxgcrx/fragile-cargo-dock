# Training Feedback

## Final-policy outcome
score=-6.735980, len=399.000000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-102.591639, -0.725095]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| joint_dock_completion | 87.712652 | 98.6% | 98.6% | 100.0% |
| boundary_health_penalty | -1.254612 | -1.4% | 1.4% | 1.6% |
| dock_approach_improvement | 0.000000 | 0.0% | 0.0% | 0.0% |
| fragile_impact_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
