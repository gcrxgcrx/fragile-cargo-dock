# Training Feedback

## Final-policy outcome
score=233.812889, len=223.500000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[3.812245, 310.066789]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 166.000000 | 96.1% | 96.1% | 3.7% |
| crate_progress | 3.990821 | 2.3% | 2.3% | 64.7% |
| cart_approach | 1.248772 | 0.7% | 0.9% | 99.2% |
| shove | -1.150888 | -0.7% | 0.7% | 17.9% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
