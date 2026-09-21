# Training Feedback

## Final-policy outcome
score=19.685928, len=396.850000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[3.561715, 309.050327]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 19.000000 | 74.9% | 74.9% | 0.2% |
| crate_progress | 3.735658 | 14.7% | 16.4% | 87.6% |
| cart_approach | 1.240776 | 4.9% | 7.6% | 99.4% |
| gentleness_obs | -0.169899 | -0.7% | 0.7% | 28.5% |
| shove | -0.113383 | -0.4% | 0.4% | 1.8% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
