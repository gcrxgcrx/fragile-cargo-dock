# Training Feedback

## Final-policy outcome
score=4.152159, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[3.573611, 5.268676]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_align | 269.390860 | 62.6% | 62.6% | 56.5% |
| soft_contact | -98.245238 | -22.8% | 22.8% | 36.7% |
| crate_progress | 46.497495 | 10.8% | 10.8% | 80.6% |
| crate_speed_pen | -12.627522 | -2.9% | 2.9% | 14.7% |
| action_smooth | -3.773961 | -0.9% | 0.9% | 100.0% |
| dock_event | 0.000000 | 0.0% | 0.0% | 0.0% |
| out_of_bounds | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
