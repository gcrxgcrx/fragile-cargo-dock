# Training Feedback

## Final-policy outcome
score=-9.618764, len=388.000000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-102.494775, 2.404809]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_docking_quality | 233.031910 | 92.6% | 92.6% | 100.0% |
| soft_contact_penalty | -6.895766 | -2.7% | 2.7% | 39.1% |
| action_smoothness | -6.702027 | -2.7% | 2.7% | 100.0% |
| crate_to_dock_progress | 1.327471 | 0.5% | 1.0% | 46.6% |
| out_of_bounds_penalty | -1.607394 | -0.6% | 0.6% | 1.0% |
| crate_speed_penalty_near_dock | -0.946177 | -0.4% | 0.4% | 45.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
