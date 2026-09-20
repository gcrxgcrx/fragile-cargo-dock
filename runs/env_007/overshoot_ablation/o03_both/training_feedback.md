# Training Feedback

## Final-policy outcome
score=-1.001157, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-1.865160, 0.413098]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_to_dock_progress | 0.492152 | 57.8% | 57.8% | 1.5% |
| dock_speed_penalty | -0.218902 | -25.7% | 25.7% | 0.7% |
| soft_contact | -0.134502 | -15.8% | 15.8% | 0.2% |
| REPAIR_gentleness | -0.005499 | -0.6% | 0.6% | 0.2% |
| OVERSHOOT_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
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
