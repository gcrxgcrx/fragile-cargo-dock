# Training Feedback

## Final-policy outcome
score=3.351513, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-0.827529, 4.549198]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| joint_completion_gate | 27.346523 | 64.8% | 64.8% | 74.9% |
| crate_to_dock_progress | 9.677630 | 22.9% | 30.3% | 41.1% |
| soft_contact_penalty | -2.001159 | -4.7% | 4.7% | 14.0% |
| boundary_avoidance | -0.039356 | -0.1% | 0.1% | 0.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
