# Training Feedback

## Final-policy outcome
score=-13.403393, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-46.679837, 25.776838]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_penalty | -20.744399 | -82.3% | 82.3% | 100.0% |
| progress | 2.584108 | 10.3% | 12.3% | 100.0% |
| orientation_penalty | -1.360607 | -5.4% | 5.4% | 100.0% |
| landing_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
