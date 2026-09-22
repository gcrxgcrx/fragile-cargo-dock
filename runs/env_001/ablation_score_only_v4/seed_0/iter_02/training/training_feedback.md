# Training Feedback

## Final-policy outcome
score=82.567303, len=992.650000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[47.912575, 188.381457]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_gate | 636.621547 | 99.7% | 99.7% | 99.6% |
| progress_reward | 1.365893 | 0.2% | 0.2% | 100.0% |
| velocity_penalty | -0.172857 | -0.0% | 0.0% | 1.6% |
| angular_penalty | -0.152699 | -0.0% | 0.0% | 82.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
