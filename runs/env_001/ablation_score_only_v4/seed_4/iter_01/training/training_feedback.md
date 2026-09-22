# Training Feedback

## Final-policy outcome
score=-11.794833, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-45.243172, 22.258570]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 1781.539505 | 98.9% | 98.9% | 100.0% |
| safe_landing_penalty | -14.344730 | -0.8% | 0.8% | 100.0% |
| fuel_penalty | -6.137000 | -0.3% | 0.3% | 61.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
