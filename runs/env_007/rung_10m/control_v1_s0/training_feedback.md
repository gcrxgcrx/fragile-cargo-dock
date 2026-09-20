# Training Feedback

## Final-policy outcome
score=68.785673, len=386.650000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[3.883581, 309.642832]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 238.000000 | 97.1% | 97.1% | 3.1% |
| crate_progress | 3.804071 | 1.6% | 1.8% | 90.5% |
| cart_approach | 1.313829 | 0.5% | 0.8% | 99.4% |
| shove | -0.805843 | -0.3% | 0.3% | 8.7% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
