# Training Feedback

## Final-policy outcome
score=-1.200747, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-1.803801, 0.769675]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_proximity | 67.027635 | 99.0% | 99.0% | 100.0% |
| action_smoothness | -0.641775 | -0.9% | 0.9% | 100.0% |
| crate_progress | 0.026746 | 0.0% | 0.0% | 1.5% |
| crate_docking_quality | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_proximity_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| out_of_bounds_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| soft_contact_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| speed_near_dock_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
