# Training Feedback

## Final-policy outcome
score=-102.604528, len=109.850000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-103.865828, -100.553366]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| action_smoothness | -2.814708 | -53.3% | 53.3% | 100.0% |
| out_of_bounds_penalty | -2.468592 | -46.7% | 46.7% | 10.2% |
| crate_docking_quality | 0.000000 | 0.0% | 0.0% | 0.0% |
| crate_speed_penalty_near_dock | 0.000000 | 0.0% | 0.0% | 0.0% |
| crate_to_dock_progress | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| soft_contact_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
