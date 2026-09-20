# Training Feedback

## Final-policy outcome
score=0.081822, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-1.620976, 2.274452]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 3.070961 | 50.3% | 50.3% | 15.4% |
| gentleness | -3.030330 | -49.7% | 49.7% | 6.2% |
| first_enter | 0.000000 | 0.0% | 0.0% | 0.0% |
| near_dock_speed_pen | 0.000000 | 0.0% | 0.0% | 0.0% |
| oob_pen | 0.000000 | 0.0% | 0.0% | 0.0% |
| settle | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
