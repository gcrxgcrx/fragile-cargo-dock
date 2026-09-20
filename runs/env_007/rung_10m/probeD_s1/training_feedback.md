# Training Feedback

## Final-policy outcome
score=143.543443, len=305.550000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[0.999545, 310.545027]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 111.000000 | 93.7% | 93.7% | 1.8% |
| crate_progress | 3.637061 | 3.1% | 3.4% | 81.6% |
| cart_approach | 1.242046 | 1.0% | 1.8% | 99.5% |
| shove | -1.148087 | -1.0% | 1.0% | 13.0% |
| gentleness_obs | -0.078461 | -0.1% | 0.1% | 17.9% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
