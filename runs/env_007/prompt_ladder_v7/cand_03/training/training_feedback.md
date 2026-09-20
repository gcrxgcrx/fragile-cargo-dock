# Training Feedback

## Final-policy outcome
score=0.034507, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-1.990011, 1.510271]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| push_toward_dock | 0.065498 | 81.0% | 81.0% | 0.2% |
| gentleness | -0.015349 | -19.0% | 19.0% | 0.4% |
| crate_to_dock_progress | 0.000000 | 0.0% | 0.0% | 0.0% |
| first_entry_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |
| out_of_bounds_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| settled_per_step | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
