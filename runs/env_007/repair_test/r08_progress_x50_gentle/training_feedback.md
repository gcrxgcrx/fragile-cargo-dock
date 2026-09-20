# Training Feedback

## Final-policy outcome
score=-3.190769, len=399.950000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-103.431427, 4.549636]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_to_dock_progress | 98.756575 | 69.7% | 69.7% | 45.5% |
| dock_speed_penalty | -39.367593 | -27.8% | 27.8% | 24.9% |
| bounds_guard | -2.492612 | -1.8% | 1.8% | 0.5% |
| soft_contact | -0.990191 | -0.7% | 0.7% | 6.8% |
| REPAIR_gentleness | -0.040625 | -0.0% | 0.0% | 6.7% |
| enter_dock_event | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
