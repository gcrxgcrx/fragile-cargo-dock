# Training Feedback

## Final-policy outcome
score=143.164801, len=363.550000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[1.575055, 309.772053]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| REPAIR_settled_stream | 440.000000 | 40.5% | 40.5% | 6.1% |
| crate_to_dock_progress | 401.285458 | 36.9% | 36.9% | 63.5% |
| success_event | 135.000000 | 12.4% | 12.4% | 0.1% |
| dock_speed_penalty | -84.613410 | -7.8% | 7.8% | 65.2% |
| enter_dock_event | 25.500000 | 2.3% | 2.3% | 0.2% |
| soft_contact | -1.095938 | -0.1% | 0.1% | 13.7% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
