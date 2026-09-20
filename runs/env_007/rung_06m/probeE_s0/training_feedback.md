# Training Feedback

## Final-policy outcome
score=6.997913, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[2.334906, 9.795045]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_progress | 3.072257 | 27.9% | 43.9% | 65.0% |
| dock_enter | 4.000000 | 36.4% | 36.4% | 0.2% |
| cart_approach | 1.014534 | 9.2% | 17.7% | 98.3% |
| gentleness | -0.223616 | -2.0% | 2.0% | 14.6% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
