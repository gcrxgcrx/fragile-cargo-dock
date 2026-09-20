# Training Feedback

## Final-policy outcome
score=7.170008, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[2.777325, 8.708189]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_progress | 2.844139 | 44.5% | 76.3% | 88.2% |
| cart_approach | 1.313443 | 20.5% | 23.7% | 90.8% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_settled_hold | 0.000000 | 0.0% | 0.0% | 0.0% |
| shove | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
