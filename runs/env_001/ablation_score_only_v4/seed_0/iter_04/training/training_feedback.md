# Training Feedback

## Final-policy outcome
score=185.334981, len=760.200000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[115.069809, 274.682086]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| braking_gate | 438.143134 | 66.1% | 66.1% | 100.0% |
| landing_reward | 222.823710 | 33.6% | 33.6% | 72.2% |
| progress_reward | 1.290232 | 0.2% | 0.2% | 99.8% |
| velocity_penalty | -0.454286 | -0.1% | 0.1% | 5.4% |
| angular_penalty | -0.253567 | -0.0% | 0.0% | 68.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
