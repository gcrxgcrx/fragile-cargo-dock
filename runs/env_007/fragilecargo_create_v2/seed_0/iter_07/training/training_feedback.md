# Training Feedback

## Final-policy outcome
score=1.973671, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[0.787302, 2.556891]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| joint_completion | 3892.828131 | 99.9% | 99.9% | 76.1% |
| crate_to_dock_progress | 3.085774 | 0.1% | 0.1% | 39.8% |
| soft_contact_penalty | -1.580665 | -0.0% | 0.0% | 12.8% |
| boundary_avoidance | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
