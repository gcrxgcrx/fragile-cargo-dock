# Training Feedback

## Final-policy outcome
score=-73.970054, len=223.600000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-93.401416, 3.835937]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_progress | 1.556671 | 13.6% | 51.2% | 61.4% |
| dock_enter | 4.000000 | 35.0% | 35.0% | 0.4% |
| cart_approach | 1.200824 | 10.5% | 13.1% | 99.1% |
| gentleness | -0.064894 | -0.6% | 0.6% | 36.2% |
| boundary | -0.011569 | -0.1% | 0.1% | 0.2% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
