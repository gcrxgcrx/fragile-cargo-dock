# Training Feedback

## Final-policy outcome
score=3.454828, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[2.263830, 4.539420]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_inside_alignment | 690.687618 | 26.9% | 26.9% | 66.9% |
| dock_proximity | 617.994681 | 24.1% | 24.1% | 100.0% |
| dock_inside_settling | 572.615100 | 22.3% | 22.3% | 66.0% |
| joint_condition_proxy | 502.675113 | 19.6% | 19.6% | 65.1% |
| contact_gated_push | 141.729644 | 5.5% | 5.5% | 18.0% |
| crate_to_dock_progress | 23.063571 | 0.9% | 0.9% | 41.1% |
| impact_softness | -10.594933 | -0.4% | 0.4% | 17.4% |
| action_smoothness | -4.683290 | -0.2% | 0.2% | 100.0% |
| boundary_avoidance | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
