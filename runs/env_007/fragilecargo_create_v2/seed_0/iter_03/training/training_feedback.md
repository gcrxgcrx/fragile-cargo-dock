# Training Feedback

## Final-policy outcome
score=2.126229, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[0.227769, 3.178838]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| joint_completion_proxy | 696.454021 | 98.9% | 98.9% | 73.2% |
| crate_to_dock_progress | 5.479223 | 0.8% | 0.9% | 42.5% |
| soft_contact_penalty | -1.454519 | -0.2% | 0.2% | 10.7% |
| boundary_avoidance | -0.001092 | -0.0% | 0.0% | 0.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
