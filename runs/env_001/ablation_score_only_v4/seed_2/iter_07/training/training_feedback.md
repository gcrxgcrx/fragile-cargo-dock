# Training Feedback

## Final-policy outcome
score=13.235179, len=971.250000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-51.159645, 177.259542]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 20.402363 | 74.9% | 74.9% | 2.2% |
| progress | 2.533314 | 9.3% | 11.5% | 99.9% |
| orientation_penalty | -2.017821 | -7.4% | 7.4% | 100.0% |
| velocity_penalty | -1.686128 | -6.2% | 6.2% | 94.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
