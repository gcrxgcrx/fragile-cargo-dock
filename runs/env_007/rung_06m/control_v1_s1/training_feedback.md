# Training Feedback

## Final-policy outcome
score=1.501054, len=392.600000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-93.344856, 9.296137]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| cart_approach | 0.787946 | 8.1% | 48.0% | 96.5% |
| crate_progress | 3.290937 | 34.0% | 45.9% | 51.7% |
| shove | -0.593288 | -6.1% | 6.1% | 6.9% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_settled_hold | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
