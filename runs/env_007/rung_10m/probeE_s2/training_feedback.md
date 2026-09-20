# Training Feedback

## Final-policy outcome
score=126.256109, len=343.950000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[0.587508, 310.478885]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_event | 120.000000 | 93.9% | 93.9% | 0.1% |
| crate_progress | 3.665359 | 2.9% | 2.9% | 63.7% |
| dock_enter | 2.250000 | 1.8% | 1.8% | 0.1% |
| cart_approach | 1.222271 | 1.0% | 1.3% | 98.7% |
| gentleness | -0.173727 | -0.1% | 0.1% | 15.7% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
