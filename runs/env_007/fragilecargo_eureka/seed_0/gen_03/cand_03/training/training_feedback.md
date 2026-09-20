# Training Feedback

## Final-policy outcome
score=2.418648, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[1.979255, 3.521921]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_gated_push | 128.680315 | 29.0% | 29.0% | 32.9% |
| crate_to_dock_progress | 118.554289 | 26.8% | 26.8% | 51.0% |
| joint_completion_proxy | 103.138179 | 23.3% | 23.3% | 65.6% |
| crate_dock_alignment | 73.312681 | 16.5% | 16.5% | 99.6% |
| crate_settling | -18.694804 | -4.2% | 4.2% | 41.0% |
| impact_softness | -0.631147 | -0.1% | 0.1% | 0.2% |
| boundary_avoidance | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
