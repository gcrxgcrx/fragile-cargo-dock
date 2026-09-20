# Training Feedback

## Final-policy outcome
score=18.669944, len=397.400000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-1.780549, 309.249078]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| docking_settle_success | 390.251048 | 98.5% | 98.5% | 41.4% |
| crate_progress_toward_dock | 5.070012 | 1.3% | 1.4% | 64.7% |
| fragile_handling_guard | -0.035035 | -0.0% | 0.0% | 0.1% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
