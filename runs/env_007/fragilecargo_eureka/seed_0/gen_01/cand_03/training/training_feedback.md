# Training Feedback

## Final-policy outcome
score=3.549992, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[2.300514, 4.136332]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_dock_alignment | 193.084150 | 31.0% | 31.0% | 95.7% |
| contact_gated_push | 147.933115 | 23.7% | 23.7% | 28.0% |
| joint_completion_proxy | 101.223282 | 16.2% | 16.2% | 69.7% |
| crate_to_dock_progress | 93.169301 | 14.9% | 14.9% | 47.4% |
| crate_settling | -88.090177 | -14.1% | 14.1% | 37.5% |
| impact_softness | -0.186472 | -0.0% | 0.0% | 0.2% |
| boundary_avoidance | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
