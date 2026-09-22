# Training Feedback

## Final-policy outcome
score=202.180509, len=468.800000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-53.435261, 255.319283]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 325.402169 | 72.8% | 72.8% | 100.0% |
| landing_success | 118.591126 | 26.5% | 26.5% | 15.4% |
| velocity_damping | -2.297143 | -0.5% | 0.5% | 99.8% |
| angle_penalty | -0.405612 | -0.1% | 0.1% | 100.0% |
| angvel_penalty | -0.255753 | -0.1% | 0.1% | 96.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
