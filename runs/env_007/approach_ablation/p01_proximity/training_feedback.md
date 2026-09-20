# Training Feedback

## Final-policy outcome
score=3.945284, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[0.481515, 5.134761]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| APPROACH_proximity | 1953.875824 | 86.9% | 86.9% | 100.0% |
| crate_to_dock_progress | 195.647250 | 8.7% | 8.7% | 42.8% |
| dock_speed_penalty | -96.539026 | -4.3% | 4.3% | 37.9% |
| soft_contact | -2.337893 | -0.1% | 0.1% | 25.0% |
| REPAIR_settled_stream | 0.000000 | 0.0% | 0.0% | 0.0% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| enter_dock_event | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
