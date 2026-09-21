# Training Feedback

## Final-policy outcome
score=52.536776, len=370.300000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[2.676243, 309.891425]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 97.000000 | 92.5% | 92.5% | 1.3% |
| crate_progress | 3.582994 | 3.4% | 4.1% | 85.0% |
| cart_approach | 1.203016 | 1.1% | 2.3% | 99.6% |
| shove | -1.094456 | -1.0% | 1.0% | 10.6% |
| gentleness_obs | -0.071564 | -0.1% | 0.1% | 12.0% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
