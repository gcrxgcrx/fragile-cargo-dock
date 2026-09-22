# Training Feedback

## Final-policy outcome
score=279.630213, len=194.900000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[8.607367, 310.627001]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 270.000000 | 96.0% | 96.0% | 0.5% |
| dock_enter | 5.000000 | 1.8% | 1.8% | 0.5% |
| progress | 4.065187 | 1.4% | 1.5% | 71.7% |
| approach_cargo | 1.199355 | 0.4% | 0.6% | 99.3% |
| time_cost | -0.389800 | -0.1% | 0.1% | 100.0% |
| action_cost | -0.061829 | -0.0% | 0.0% | 100.0% |
| roughness | -0.029730 | -0.0% | 0.0% | 16.5% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
