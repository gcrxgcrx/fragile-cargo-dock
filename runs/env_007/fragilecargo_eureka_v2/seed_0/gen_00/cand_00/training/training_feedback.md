# Training Feedback

## Final-policy outcome
score=2.341422, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[1.651779, 7.975620]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| joint_completion_proxy | 731.008292 | 51.9% | 51.9% | 63.0% |
| crate_dock_alignment | 349.121181 | 24.8% | 24.8% | 75.9% |
| crate_settling | 302.020380 | 21.5% | 21.5% | 63.0% |
| crate_to_dock_progress | 22.297961 | 1.6% | 1.8% | 44.7% |
| gentle_contact | -0.013375 | -0.0% | 0.0% | 0.0% |
| boundary_avoidance | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_proximity | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
