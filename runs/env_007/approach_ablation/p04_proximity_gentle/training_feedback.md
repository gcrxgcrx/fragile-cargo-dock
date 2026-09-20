# Training Feedback

## Final-policy outcome
score=3.604665, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[1.478286, 8.546050]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| APPROACH_proximity | 391.585489 | 55.9% | 55.9% | 100.0% |
| crate_to_dock_progress | 183.178822 | 26.2% | 26.2% | 34.8% |
| dock_speed_penalty | -119.264114 | -17.0% | 17.0% | 48.1% |
| enter_dock_event | 3.000000 | 0.4% | 0.4% | 0.0% |
| soft_contact | -2.979869 | -0.4% | 0.4% | 20.5% |
| REPAIR_gentleness | -0.124421 | -0.0% | 0.0% | 20.5% |
| REPAIR_settled_stream | 0.000000 | 0.0% | 0.0% | 0.0% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
