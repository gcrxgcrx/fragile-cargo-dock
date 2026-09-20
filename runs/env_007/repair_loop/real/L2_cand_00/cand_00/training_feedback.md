# Training Feedback

## Final-policy outcome
score=-7.378563, len=390.000000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-102.512692, -0.329807]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| docking_quality | 282.043724 | 74.7% | 74.7% | 99.9% |
| crate_to_dock_progress | 95.704095 | 25.3% | 25.3% | 99.9% |
| dock_speed_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| enter_dock_event | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| soft_contact | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
