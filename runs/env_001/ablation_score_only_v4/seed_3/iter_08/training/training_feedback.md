# Training Feedback

## Final-policy outcome
score=-39.402011, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-76.591527, 4.213115]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_shaping | 1585.527597 | 82.0% | 82.0% | 100.0% |
| approach_and_land | 296.742560 | 15.3% | 15.3% | 100.0% |
| efficiency_penalty | -37.615312 | -1.9% | 1.9% | 100.0% |
| progress_reward | 10.085865 | 0.5% | 0.7% | 100.0% |
| orientation_penalty | -0.979253 | -0.1% | 0.1% | 100.0% |
| velocity_constraint | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
