# Training Feedback

## Final-policy outcome
score=-7.841064, len=386.600000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-102.953987, 5.976647]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| joint_completion_gate | 205.294032 | 95.4% | 95.4% | 76.6% |
| crate_to_dock_progress | 7.044770 | 3.3% | 3.8% | 40.7% |
| soft_contact_penalty | -1.658561 | -0.8% | 0.8% | 12.8% |
| boundary_avoidance | -0.061118 | -0.0% | 0.0% | 0.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
