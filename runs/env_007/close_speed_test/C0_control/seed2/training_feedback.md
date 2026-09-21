# Training Feedback

## Final-policy outcome
score=263.068502, len=203.500000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-0.810984, 309.837864]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 170.000000 | 96.1% | 96.1% | 4.2% |
| crate_progress | 3.611695 | 2.0% | 2.2% | 69.3% |
| cart_approach | 0.854360 | 0.5% | 1.1% | 98.9% |
| shove | -1.201642 | -0.7% | 0.7% | 18.7% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
