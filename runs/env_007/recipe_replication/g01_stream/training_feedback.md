# Training Feedback

## Final-policy outcome
score=-1.287342, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-2.000232, 0.003489]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_dock_progress | 0.816701 | 100.0% | 100.0% | 0.2% |
| REPAIR_settled_stream | 0.000000 | 0.0% | 0.0% | 0.0% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_quality_gate | 0.000000 | 0.0% | 0.0% | 0.0% |
| enter_event | 0.000000 | 0.0% | 0.0% | 0.0% |
| gentleness | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
