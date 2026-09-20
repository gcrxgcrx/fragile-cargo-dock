# Training Feedback

## Final-policy outcome
score=-3.440704, len=388.700000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-103.395582, 5.728433]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_progress_toward_dock | 5.211910 | 35.5% | 53.9% | 43.4% |
| push_intensity_guard | -5.660105 | -38.6% | 38.6% | 19.6% |
| docking_settle_success | 0.422161 | 2.9% | 2.9% | 0.5% |
| fragile_contact_guard | -0.326004 | -2.2% | 2.2% | 0.3% |
| steering_effort | -0.226035 | -1.5% | 1.5% | 100.0% |
| floor_bounds_guard | -0.123887 | -0.8% | 0.8% | 0.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
