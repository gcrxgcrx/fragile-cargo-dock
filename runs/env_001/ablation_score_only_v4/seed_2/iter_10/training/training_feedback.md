# Training Feedback

## Final-policy outcome
score=-433.260728, len=114.950000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-544.498076, -341.298169]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 0.750616 | 65.2% | 98.7% | 100.0% |
| velocity_penalty | -0.009888 | -0.9% | 0.9% | 1.3% |
| orientation_penalty | -0.005427 | -0.5% | 0.5% | 1.3% |
| landing_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 18/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
