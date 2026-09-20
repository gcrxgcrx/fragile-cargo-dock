# Training Feedback

## Final-policy outcome
score=218.081514, len=238.650000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[0.484600, 309.914251]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 140.000000 | 95.8% | 95.8% | 2.9% |
| crate_progress | 3.799937 | 2.6% | 2.7% | 66.5% |
| cart_approach | 1.015140 | 0.7% | 1.2% | 99.1% |
| shove | -0.373384 | -0.3% | 0.3% | 9.8% |
| gentleness_obs | -0.183560 | -0.1% | 0.1% | 16.0% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
