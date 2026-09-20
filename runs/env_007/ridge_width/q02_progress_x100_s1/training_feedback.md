# Training Feedback

## Final-policy outcome
score=233.980005, len=237.350000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[2.817426, 310.417974]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_to_dock_progress | 434.617083 | 45.8% | 45.8% | 48.1% |
| success_event | 225.000000 | 23.7% | 23.7% | 0.3% |
| REPAIR_settled_stream | 150.000000 | 15.8% | 15.8% | 3.2% |
| dock_speed_penalty | -107.972359 | -11.4% | 11.4% | 52.6% |
| enter_dock_event | 28.500000 | 3.0% | 3.0% | 0.4% |
| soft_contact | -2.143948 | -0.2% | 0.2% | 20.9% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
