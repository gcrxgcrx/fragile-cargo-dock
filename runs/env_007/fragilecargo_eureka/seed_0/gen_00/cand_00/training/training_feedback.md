# Training Feedback

## Final-policy outcome
score=-102.240936, len=179.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-102.832904, -100.664939]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_dock_alignment | 9.926713 | 66.4% | 66.4% | 100.0% |
| time_efficiency | -2.531405 | -16.9% | 16.9% | 99.4% |
| boundary_avoidance | -2.485857 | -16.6% | 16.6% | 8.7% |
| contact_gated_push | 0.000000 | 0.0% | 0.0% | 0.0% |
| crate_settling | 0.000000 | 0.0% | 0.0% | 0.0% |
| crate_to_dock_progress | 0.000000 | 0.0% | 0.0% | 0.0% |
| impact_softness | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 4/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
