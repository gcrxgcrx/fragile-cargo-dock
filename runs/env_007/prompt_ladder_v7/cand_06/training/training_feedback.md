# Training Feedback

## Final-policy outcome
score=-0.268560, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-1.917738, 1.310026]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gentleness | -2.189042 | -58.2% | 58.2% | 2.1% |
| progress_delta | 1.571238 | 41.8% | 41.8% | 5.4% |
| bound_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| enter_event | 0.000000 | 0.0% | 0.0% | 0.0% |
| settle | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
