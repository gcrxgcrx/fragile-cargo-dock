# Training Feedback

## Final-policy outcome
score=33.742238, len=379.300000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[2.780103, 309.234392]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| docking_settle | 351.467852 | 89.9% | 89.9% | 43.3% |
| crate_progress_toward_dock | 36.882233 | 9.4% | 9.9% | 46.7% |
| fragile_handling_penalty | -1.015487 | -0.3% | 0.3% | 4.8% |
| boundary_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
