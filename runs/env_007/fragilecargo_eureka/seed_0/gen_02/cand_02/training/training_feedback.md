# Training Feedback

## Final-policy outcome
score=3.376906, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[1.724166, 7.889897]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_dock_alignment | 229.173066 | 33.1% | 33.1% | 96.7% |
| crate_to_dock_progress | 157.247320 | 22.7% | 22.7% | 44.3% |
| joint_completion_proxy | 126.821871 | 18.3% | 18.3% | 74.4% |
| contact_gated_push | 119.838468 | 17.3% | 17.3% | 22.1% |
| crate_settling | -59.560120 | -8.6% | 8.6% | 34.3% |
| impact_softness | -0.064871 | -0.0% | 0.0% | 0.1% |
| boundary_avoidance | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
