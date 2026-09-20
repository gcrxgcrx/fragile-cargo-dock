# Training Feedback

## Final-policy outcome
score=4.863917, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[3.566066, 9.317008]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_progress_delta | 31.414969 | 84.8% | 84.8% | 88.3% |
| joint_completion_gate | 4.766689 | 12.9% | 12.9% | 88.3% |
| local_obstacle_penalty | -0.599346 | -1.6% | 1.6% | 1.1% |
| fragile_impact_penalty | -0.243437 | -0.7% | 0.7% | 3.4% |
| crate_speed_hinge | -0.015270 | -0.0% | 0.0% | 0.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
