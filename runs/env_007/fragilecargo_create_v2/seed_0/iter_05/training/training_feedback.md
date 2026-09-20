# Training Feedback

## Final-policy outcome
score=4.175785, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-0.921406, 7.759889]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| joint_completion | 603.973623 | 98.2% | 98.2% | 71.9% |
| crate_to_dock_progress | 8.452845 | 1.4% | 1.5% | 41.6% |
| soft_contact_penalty | -1.487839 | -0.2% | 0.2% | 11.3% |
| boundary_avoidance | -0.002459 | -0.0% | 0.0% | 0.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
