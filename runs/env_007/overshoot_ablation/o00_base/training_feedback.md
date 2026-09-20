# Training Feedback

## Final-policy outcome
score=144.373816, len=371.900000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[8.726330, 309.621255]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| REPAIR_settled_stream | 680.000000 | 58.3% | 58.3% | 9.1% |
| crate_to_dock_progress | 228.528243 | 19.6% | 19.6% | 69.6% |
| success_event | 135.000000 | 11.6% | 11.6% | 0.1% |
| dock_speed_penalty | -90.238859 | -7.7% | 7.7% | 67.0% |
| enter_dock_event | 30.000000 | 2.6% | 2.6% | 0.3% |
| soft_contact | -2.234459 | -0.2% | 0.2% | 18.1% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
