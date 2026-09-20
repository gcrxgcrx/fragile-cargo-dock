# Training Feedback

## Final-policy outcome
score=-20.517013, len=363.400000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[-104.210661, 3.181410]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_progress | 1.909625 | 33.7% | 37.4% | 45.0% |
| contact_pen | -1.940532 | -34.3% | 34.3% | 10.4% |
| align_term | 1.061824 | 18.8% | 18.8% | 2.4% |
| smooth_pen | -0.402525 | -7.1% | 7.1% | 100.0% |
| overspeed_pen | -0.138828 | -2.5% | 2.5% | 0.4% |
| dock_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |
| oob_pen | 0.000000 | 0.0% | 0.0% | 0.0% |
| speed_term | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
