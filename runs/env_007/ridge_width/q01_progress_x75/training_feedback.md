# Training Feedback

## Final-policy outcome
score=4.318743, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[3.812821, 9.057080]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_to_dock_progress | 280.826299 | 79.1% | 79.1% | 85.3% |
| dock_speed_penalty | -71.204423 | -20.0% | 20.0% | 71.4% |
| soft_contact | -1.615683 | -0.5% | 0.5% | 28.3% |
| enter_dock_event | 1.500000 | 0.4% | 0.4% | 0.0% |
| REPAIR_settled_stream | 0.000000 | 0.0% | 0.0% | 0.0% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
