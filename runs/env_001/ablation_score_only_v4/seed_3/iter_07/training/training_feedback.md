# Training Feedback

## Final-policy outcome
score=-37.653007, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-73.551976, -2.655894]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_shaping | 1591.730059 | 95.4% | 95.4% | 100.0% |
| fuel_penalty | -62.651000 | -3.8% | 3.8% | 90.7% |
| progress_reward | 9.996644 | 0.6% | 0.8% | 100.0% |
| orientation_penalty | -1.094045 | -0.1% | 0.1% | 100.0% |
| landing_quality | 0.000000 | 0.0% | 0.0% | 0.0% |
| velocity_constraint | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
