# Training Feedback

## Final-policy outcome
score=-1.850232, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-3.309293, 0.555519]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_docking_quality | 332.868598 | 99.5% | 99.5% | 100.0% |
| action_smoothness | -1.517544 | -0.5% | 0.5% | 100.0% |
| crate_speed_penalty_near_dock | 0.000000 | 0.0% | 0.0% | 0.0% |
| crate_to_dock_progress | 0.000000 | 0.0% | 0.0% | 0.0% |
| out_of_bounds_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| soft_contact_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
