# Training Feedback

## Final-policy outcome
score=5.125764, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[0.587582, 8.407426]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_dock_alignment | 762.432228 | 44.3% | 44.3% | 89.6% |
| crate_settling | 613.949994 | 35.7% | 35.7% | 77.3% |
| contact_gated_push | 151.177721 | 8.8% | 8.8% | 20.1% |
| joint_condition_proxy | 145.171379 | 8.4% | 8.4% | 72.8% |
| crate_to_dock_progress | 28.587358 | 1.7% | 1.9% | 42.9% |
| impact_softness | -11.169270 | -0.6% | 0.6% | 20.1% |
| action_smoothness | -3.136361 | -0.2% | 0.2% | 100.0% |
| boundary_avoidance | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
