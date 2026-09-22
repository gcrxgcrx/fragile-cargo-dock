# Training Feedback

## Final-policy outcome
score=-72.940675, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-97.774872, -39.356460]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_shaping | 1397.054683 | 97.6% | 97.6% | 100.0% |
| efficiency_penalty | -24.650937 | -1.7% | 1.7% | 100.0% |
| progress_reward | 6.703828 | 0.5% | 0.6% | 100.0% |
| orientation_penalty | -1.110396 | -0.1% | 0.1% | 100.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |
| velocity_constraint | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
