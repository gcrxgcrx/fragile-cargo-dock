# Training Feedback

## Final-policy outcome
score=-0.676813, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-0.928184, -0.215745]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| action_penalty | -0.587490 | -100.0% | 100.0% | 100.0% |
| bounds_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| entry_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |
| gentleness | 0.000000 | 0.0% | 0.0% | 0.0% |
| progress | 0.000000 | 0.0% | 0.0% | 0.0% |
| speed_gate_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
