# Training Feedback

## Final-policy outcome
score=4.503913, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[3.575489, 8.861561]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_dock_alignment | 101.329894 | 80.8% | 80.8% | 65.0% |
| crate_settling | -12.158012 | -9.7% | 9.7% | 14.6% |
| soft_contact_penalty | -8.430463 | -6.7% | 6.7% | 16.6% |
| crate_to_dock_progress | 3.535816 | 2.8% | 2.8% | 45.8% |
| boundary_avoidance | -0.002291 | -0.0% | 0.0% | 0.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
