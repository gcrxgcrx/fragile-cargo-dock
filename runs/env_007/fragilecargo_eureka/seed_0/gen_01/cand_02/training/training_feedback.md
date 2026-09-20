# Training Feedback

## Final-policy outcome
score=-1.339665, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-2.860574, 1.097786]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| joint_condition_proxy | 428.465073 | 99.8% | 99.8% | 100.0% |
| action_smoothness | -0.962226 | -0.2% | 0.2% | 100.0% |
| boundary_avoidance | -0.043716 | -0.0% | 0.0% | 1.0% |
| contact_gated_push | 0.000000 | 0.0% | 0.0% | 0.0% |
| crate_dock_alignment | 0.000000 | 0.0% | 0.0% | 0.0% |
| crate_settling | 0.000000 | 0.0% | 0.0% | 0.0% |
| crate_to_dock_progress | 0.000000 | 0.0% | 0.0% | 0.0% |
| impact_softness | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
