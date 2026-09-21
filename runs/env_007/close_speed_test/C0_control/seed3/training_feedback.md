# Training Feedback

## Final-policy outcome
score=171.996540, len=366.100000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[3.815162, 310.223694]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 191.000000 | 97.1% | 97.1% | 2.6% |
| crate_progress | 3.979458 | 2.0% | 2.0% | 84.9% |
| cart_approach | 1.231932 | 0.6% | 0.9% | 98.3% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| shove | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
