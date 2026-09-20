# Training Feedback

## Final-policy outcome
score=294.226567, len=236.050000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[3.904070, 310.501655]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| native_passthrough | 294.226567 | 99.9% | 100.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
