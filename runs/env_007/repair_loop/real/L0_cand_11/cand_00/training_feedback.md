# Training Feedback

## Final-policy outcome
score=-2.932651, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-4.056204, 0.206189]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_docking_quality | 535.894130 | 99.8% | 99.8% | 100.0% |
| out_of_bounds_penalty | -0.435364 | -0.1% | 0.1% | 3.1% |
| action_smoothness_penalty | -0.415043 | -0.1% | 0.1% | 100.0% |
| crate_to_dock_progress | 0.180854 | 0.0% | 0.0% | 1.7% |
| crate_push_signal | 0.010287 | 0.0% | 0.0% | 0.7% |
| crate_speed_penalty_near_dock | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_proximity_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| soft_contact_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
