# Training Feedback

## Final-policy outcome
score=-1.372462, len=392.300000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-98.540211, 8.501904]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_progress | 2.663237 | 38.2% | 64.9% | 56.9% |
| cart_approach | 1.088880 | 15.6% | 25.9% | 98.7% |
| shove | -0.644848 | -9.2% | 9.2% | 9.2% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_settled_hold | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
