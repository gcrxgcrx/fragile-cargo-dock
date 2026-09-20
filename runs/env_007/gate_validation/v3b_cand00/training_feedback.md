# Training Feedback

## Final-policy outcome
score=-102.914280, len=148.650000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-104.059630, -100.673527]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| smoothness | -0.446754 | -52.1% | 52.1% | 100.0% |
| edge_penalty | -0.410489 | -47.9% | 47.9% | 6.5% |
| align_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |
| entered_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |
| gentleness | 0.000000 | 0.0% | 0.0% | 0.0% |
| progress | 0.000000 | 0.0% | 0.0% | 0.0% |
| speed_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 12/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
