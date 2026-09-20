# Training Feedback

## Final-policy outcome
score=5.740724, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[2.662217, 9.482641]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_settle_and_align | 181.268445 | 94.6% | 94.6% | 71.6% |
| crate_to_dock_progress | 9.613547 | 5.0% | 5.4% | 23.8% |
| fragile_impact_avoidance | -0.013040 | -0.0% | 0.0% | 0.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
