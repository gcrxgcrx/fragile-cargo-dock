# Training Feedback

## Final-policy outcome
score=-77.465178, len=908.050000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[-258.898766, 85.416659]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 80.426249 | 93.0% | 93.0% | 1.4% |
| progress | 1.563803 | 1.8% | 3.7% | 100.0% |
| velocity_penalty | -1.477758 | -1.7% | 1.7% | 93.6% |
| orientation_penalty | -1.419770 | -1.6% | 1.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
