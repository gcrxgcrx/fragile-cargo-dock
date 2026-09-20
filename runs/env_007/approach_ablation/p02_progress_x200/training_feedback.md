# Training Feedback

## Final-policy outcome
score=-7.746609, len=380.600000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-98.121084, 3.367932]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_to_dock_progress | 779.154301 | 80.2% | 80.2% | 24.5% |
| dock_speed_penalty | -169.105039 | -17.4% | 17.4% | 40.9% |
| enter_dock_event | 22.500000 | 2.3% | 2.3% | 0.2% |
| soft_contact | -1.268189 | -0.1% | 0.1% | 16.3% |
| REPAIR_settled_stream | 0.000000 | 0.0% | 0.0% | 0.0% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
