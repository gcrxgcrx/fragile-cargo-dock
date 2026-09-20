# Training Feedback

## Final-policy outcome
score=5.693248, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[0.309641, 9.676386]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_to_dock_progress | 190.487447 | 57.0% | 57.0% | 80.0% |
| dock_speed_penalty | -74.155997 | -22.2% | 22.2% | 59.3% |
| REPAIR_settled_stream | 58.000000 | 17.4% | 17.4% | 0.7% |
| enter_dock_event | 10.500000 | 3.1% | 3.1% | 0.1% |
| soft_contact | -0.993758 | -0.3% | 0.3% | 29.3% |
| REPAIR_gentleness | -0.040796 | -0.0% | 0.0% | 29.2% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
