# Training Feedback

## Final-policy outcome
score=294.329988, len=183.950000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[4.908189, 309.976533]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| native_passthrough | 294.329988 | 99.8% | 100.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
