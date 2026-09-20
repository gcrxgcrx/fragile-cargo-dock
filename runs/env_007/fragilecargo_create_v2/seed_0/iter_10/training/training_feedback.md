# Training Feedback

## Final-policy outcome
score=2.929736, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[0.022902, 3.946452]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| joint_completion | 129.471428 | 88.8% | 88.8% | 100.0% |
| crate_to_dock_progress | 15.911629 | 10.9% | 11.0% | 35.6% |
| soft_contact_penalty | -0.170706 | -0.1% | 0.1% | 3.1% |
| boundary_avoidance | -0.000063 | -0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
