# Training Feedback

## Final-policy outcome
score=-2.042296, len=395.850000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-97.948417, 4.518838]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_approach | 920.511683 | 72.9% | 72.9% | 100.0% |
| crate_docking_quality | 307.984918 | 24.4% | 24.4% | 68.0% |
| crate_progress | 20.850846 | 1.7% | 1.7% | 45.5% |
| speed_near_dock_penalty | -7.137655 | -0.6% | 0.6% | 16.4% |
| out_of_bounds_penalty | -2.611964 | -0.2% | 0.2% | 0.7% |
| action_smoothness | -1.994239 | -0.2% | 0.2% | 100.0% |
| soft_contact_penalty | -0.901519 | -0.1% | 0.1% | 13.9% |
| obstacle_proximity_penalty | -0.036805 | -0.0% | 0.0% | 0.2% |
| dock_hold_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
