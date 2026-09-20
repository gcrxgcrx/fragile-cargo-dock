# Training Feedback

## Final-policy outcome
score=-67.402678, len=289.950000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[-104.393466, 0.967516]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| out_of_bounds_penalty | -1.836768 | -59.0% | 59.0% | 3.0% |
| action_smoothness | -1.278802 | -41.0% | 41.0% | 100.0% |
| crate_docking_quality | 0.000000 | 0.0% | 0.0% | 0.0% |
| crate_speed_penalty_near_dock | 0.000000 | 0.0% | 0.0% | 0.0% |
| crate_to_dock_progress | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| soft_contact_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
