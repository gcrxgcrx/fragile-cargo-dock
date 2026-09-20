# Training Feedback

## Final-policy outcome
score=-1.434554, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-3.034344, -0.179248]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 3142.876602 | 78.2% | 78.2% | 100.0% |
| dock_quality | 874.250749 | 21.7% | 21.7% | 100.0% |
| bound_pen | -2.663780 | -0.1% | 0.1% | 2.3% |
| enter_event | 0.000000 | 0.0% | 0.0% | 0.0% |
| gentleness | 0.000000 | 0.0% | 0.0% | 0.0% |
| speed_pen | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
