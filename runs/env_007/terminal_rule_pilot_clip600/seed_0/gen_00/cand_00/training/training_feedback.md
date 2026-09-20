# Training Feedback

## Final-policy outcome
score=-41.068165, len=358.750000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[-105.489415, 1.031579]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_pen | -4.512243 | -46.8% | 46.8% | 13.7% |
| crate_progress | 1.775304 | 18.4% | 23.2% | 40.0% |
| align_term | 1.308378 | 13.6% | 13.6% | 2.0% |
| overspeed_pen | -1.041473 | -10.8% | 10.8% | 0.7% |
| smooth_pen | -0.551406 | -5.7% | 5.7% | 100.0% |
| dock_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |
| oob_pen | 0.000000 | 0.0% | 0.0% | 0.0% |
| speed_term | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
