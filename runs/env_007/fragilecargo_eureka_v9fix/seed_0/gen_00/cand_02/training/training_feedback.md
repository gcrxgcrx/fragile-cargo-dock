# Training Feedback

## Final-policy outcome
score=294.178090, len=235.350000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[3.583585, 310.390296]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 285.000000 | 92.8% | 92.8% | 0.4% |
| settle_bonus | 10.500000 | 3.4% | 3.4% | 4.5% |
| dock_enter | 4.750000 | 1.5% | 1.5% | 0.4% |
| progress | 4.069580 | 1.3% | 1.3% | 75.1% |
| approach_cargo | 1.032551 | 0.3% | 0.6% | 98.6% |
| time_cost | -0.470700 | -0.2% | 0.2% | 100.0% |
| roughness | -0.336644 | -0.1% | 0.1% | 22.1% |
| action_cost | -0.064102 | -0.0% | 0.0% | 100.0% |
| boundary_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
