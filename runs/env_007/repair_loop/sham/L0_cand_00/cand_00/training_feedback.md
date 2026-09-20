# Training Feedback

## Final-policy outcome
score=-103.043781, len=165.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-104.008795, -102.107928]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| out_of_bounds_penalty | -1.529648 | -61.3% | 61.3% | 5.5% |
| action_smoothness | -0.963781 | -38.7% | 38.7% | 100.0% |
| crate_docking_quality | 0.000000 | 0.0% | 0.0% | 0.0% |
| crate_to_dock_progress | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| soft_contact_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 7/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
