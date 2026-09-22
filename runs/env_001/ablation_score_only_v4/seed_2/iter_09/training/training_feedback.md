# Training Feedback

## Final-policy outcome
score=-133.916632, len=450.200000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-203.201750, 152.022472]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 41.145376 | 88.8% | 88.8% | 1.1% |
| progress | 0.813983 | 1.8% | 6.4% | 100.0% |
| velocity_penalty | -2.171226 | -4.7% | 4.7% | 85.9% |
| orientation_penalty | -0.069515 | -0.2% | 0.2% | 85.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
