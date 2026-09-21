# Training Feedback

## Final-policy outcome
score=38.227264, len=394.050000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[3.941350, 309.797222]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 251.000000 | 96.8% | 96.8% | 3.2% |
| crate_progress | 3.567051 | 1.4% | 1.7% | 89.1% |
| cart_approach | 1.284620 | 0.5% | 1.0% | 99.8% |
| gentleness_obs | -1.063777 | -0.4% | 0.4% | 25.3% |
| shove | -0.207021 | -0.1% | 0.1% | 3.6% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
