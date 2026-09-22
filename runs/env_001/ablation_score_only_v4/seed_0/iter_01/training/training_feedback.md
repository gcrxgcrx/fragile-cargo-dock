# Training Feedback

## Final-policy outcome
score=-113.140318, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-145.529599, -65.866693]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy | 561.528617 | 99.9% | 99.9% | 99.8% |
| progress_reward | 0.160622 | 0.0% | 0.1% | 100.0% |
| velocity_penalty | -0.164667 | -0.0% | 0.0% | 1.4% |
| angular_penalty | -0.099893 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
