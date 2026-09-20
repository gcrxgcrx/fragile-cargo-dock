# Training Feedback

## Final-policy outcome
score=-20.161333, len=380.700000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[-98.053924, 7.706400]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_progress | 1.941457 | 20.4% | 63.5% | 84.1% |
| cart_approach | 1.207915 | 12.7% | 20.4% | 98.6% |
| shove | -0.910576 | -9.6% | 9.6% | 9.4% |
| gentleness_obs | -0.516034 | -5.4% | 5.4% | 29.9% |
| boundary | -0.107413 | -1.1% | 1.1% | 0.9% |
| dock_settled_hold | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
