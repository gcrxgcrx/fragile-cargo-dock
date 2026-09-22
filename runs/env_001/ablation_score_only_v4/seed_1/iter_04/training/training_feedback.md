# Training Feedback

## Final-policy outcome
score=5.515997, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-39.833641, 61.350290]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 802.097327 | 90.2% | 90.2% | 100.0% |
| landing_success | 81.722630 | 9.2% | 9.2% | 4.9% |
| velocity_damping | -4.385530 | -0.5% | 0.5% | 100.0% |
| angle_penalty | -0.402033 | -0.0% | 0.0% | 100.0% |
| angvel_penalty | -0.162602 | -0.0% | 0.0% | 96.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
