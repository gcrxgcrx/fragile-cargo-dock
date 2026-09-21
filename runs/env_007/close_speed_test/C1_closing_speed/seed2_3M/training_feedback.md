# Training Feedback

## Final-policy outcome
score=8.832895, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[8.312134, 10.028694]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 399.000000 | 97.9% | 97.9% | 5.0% |
| crate_progress | 3.760466 | 0.9% | 1.1% | 90.7% |
| cart_approach | 1.228900 | 0.3% | 0.9% | 99.5% |
| shove | -0.202230 | -0.0% | 0.0% | 3.0% |
| gentleness_obs | -0.180748 | -0.0% | 0.0% | 15.8% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
