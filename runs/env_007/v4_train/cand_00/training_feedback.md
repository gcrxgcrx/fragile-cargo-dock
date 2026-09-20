# Training Feedback

## Final-policy outcome
score=-102.792649, len=77.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-103.931624, -100.612874]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| out_of_bounds_penalty | -4.317136 | -83.8% | 83.8% | 100.0% |
| action_smoothness | -0.832822 | -16.2% | 16.2% | 100.0% |
| crate_progress | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_speed_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| enter_dock_event | 0.000000 | 0.0% | 0.0% | 0.0% |
| gentleness | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
