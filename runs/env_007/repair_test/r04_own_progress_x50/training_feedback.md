# Training Feedback

## Final-policy outcome
score=9.309147, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[8.800184, 10.423039]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_to_dock_progress | 227.641365 | 63.9% | 63.9% | 75.0% |
| dock_speed_penalty | -96.485417 | -27.1% | 27.1% | 77.6% |
| enter_dock_event | 30.000000 | 8.4% | 8.4% | 0.2% |
| soft_contact | -1.876135 | -0.5% | 0.5% | 41.0% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
