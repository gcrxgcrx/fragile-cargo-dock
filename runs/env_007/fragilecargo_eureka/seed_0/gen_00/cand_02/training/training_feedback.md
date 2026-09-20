# Training Feedback

## Final-policy outcome
score=-7.008146, len=383.700000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-102.008535, -1.104786]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_joint_condition | 288.853141 | 77.5% | 77.5% | 100.0% |
| crate_settling | 33.276177 | 8.9% | 8.9% | 100.0% |
| crate_dock_proximity | 30.017484 | 8.1% | 8.1% | 100.0% |
| crate_dock_alignment | 16.400964 | 4.4% | 4.4% | 100.0% |
| cart_boundary_avoidance | -3.715782 | -1.0% | 1.0% | 36.6% |
| action_smoothness | -0.466451 | -0.1% | 0.1% | 100.0% |
| contact_gated_push | 0.000000 | 0.0% | 0.0% | 0.0% |
| crate_boundary_avoidance | 0.000000 | 0.0% | 0.0% | 0.0% |
| crate_to_dock_progress | 0.000000 | 0.0% | 0.0% | 0.0% |
| impact_softness | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_avoidance | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
