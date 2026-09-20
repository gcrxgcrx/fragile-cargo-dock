# Training Feedback

## Final-policy outcome
score=7.394714, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[4.007258, 9.084282]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_to_dock_progress | 336.818826 | 65.0% | 65.0% | 36.3% |
| dock_speed_penalty | -113.719670 | -22.0% | 22.0% | 61.5% |
| REPAIR_settled_stream | 44.000000 | 8.5% | 8.5% | 0.5% |
| enter_dock_event | 21.000000 | 4.1% | 4.1% | 0.2% |
| soft_contact | -2.269403 | -0.4% | 0.4% | 17.0% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
