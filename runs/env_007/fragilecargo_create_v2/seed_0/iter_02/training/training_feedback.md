# Training Feedback

## Final-policy outcome
score=-102.842678, len=77.050000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-103.966218, -100.668020]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_to_dock_progress | -16.002953 | -99.0% | 99.0% | 100.0% |
| boundary_avoidance | -0.160772 | -1.0% | 1.0% | 9.3% |
| crate_dock_alignment | 0.000000 | 0.0% | 0.0% | 0.0% |
| crate_settling | 0.000000 | 0.0% | 0.0% | 0.0% |
| soft_contact_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
