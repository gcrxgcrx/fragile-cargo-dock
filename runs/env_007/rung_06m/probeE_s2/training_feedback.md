# Training Feedback

## Final-policy outcome
score=158.561735, len=292.550000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[4.041234, 310.038476]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_event | 150.000000 | 93.6% | 93.6% | 0.2% |
| dock_enter | 4.250000 | 2.7% | 2.7% | 0.3% |
| crate_progress | 3.893897 | 2.4% | 2.6% | 63.1% |
| cart_approach | 1.220470 | 0.8% | 1.0% | 98.9% |
| gentleness | -0.157705 | -0.1% | 0.1% | 17.7% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
