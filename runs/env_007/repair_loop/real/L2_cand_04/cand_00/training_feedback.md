# Training Feedback

## Final-policy outcome
score=-2.154479, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-3.480956, 0.113469]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| completion | 6801.936790 | 99.9% | 99.9% | 100.0% |
| bounds_penalty | -6.737090 | -0.1% | 0.1% | 2.7% |
| action_penalty | -0.213988 | -0.0% | 0.0% | 100.0% |
| gentleness | 0.000000 | 0.0% | 0.0% | 0.0% |
| progress | 0.000000 | 0.0% | 0.0% | 0.0% |
| speed_gate_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
