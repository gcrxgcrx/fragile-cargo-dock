# Training Feedback

## Final-policy outcome
score=7.941389, len=375.400000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-98.048456, 309.091785]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 10.000000 | 67.3% | 67.3% | 0.1% |
| crate_progress | 1.638144 | 11.0% | 18.6% | 58.2% |
| cart_approach | 1.134150 | 7.6% | 12.8% | 96.0% |
| gentleness_obs | -0.199643 | -1.3% | 1.3% | 14.4% |
| boundary | -0.000119 | -0.0% | 0.0% | 0.0% |
| shove | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
