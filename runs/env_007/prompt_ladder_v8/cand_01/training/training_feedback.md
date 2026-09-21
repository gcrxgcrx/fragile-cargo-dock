# Training Feedback

## Final-policy outcome
score=22.586606, len=391.650000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[0.644003, 310.643332]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settled_reward | 117.000000 | 76.1% | 76.1% | 1.5% |
| crate_progress | 14.035799 | 9.1% | 10.4% | 76.7% |
| success_event | 15.000000 | 9.8% | 9.8% | 0.0% |
| enter_bonus | 4.000000 | 2.6% | 2.6% | 0.2% |
| gentleness | -1.727747 | -1.1% | 1.1% | 13.7% |
| bounds_pen | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
