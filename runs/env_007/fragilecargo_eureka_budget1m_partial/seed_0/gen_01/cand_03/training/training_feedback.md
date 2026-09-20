# Training Feedback

## Final-policy outcome
score=3.686045, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[2.155558, 4.237925]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| docking_settle_success | 472.897161 | 96.6% | 96.6% | 53.2% |
| crate_progress_toward_dock | 14.643218 | 3.0% | 3.0% | 48.0% |
| approach_crate | 0.822195 | 0.2% | 0.4% | 72.8% |
| fragile_handling_guard | -0.032892 | -0.0% | 0.0% | 0.4% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
