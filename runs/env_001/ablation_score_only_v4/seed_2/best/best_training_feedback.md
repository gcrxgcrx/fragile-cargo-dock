# Training Feedback

## Final-policy outcome
score=150.409876, len=873.750000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[97.757226, 264.417707]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 430.875000 | 94.7% | 94.7% | 65.4% |
| velocity_penalty | -19.356002 | -4.3% | 4.3% | 100.0% |
| progress | 2.605007 | 0.6% | 0.6% | 100.0% |
| orientation_penalty | -1.637571 | -0.4% | 0.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
