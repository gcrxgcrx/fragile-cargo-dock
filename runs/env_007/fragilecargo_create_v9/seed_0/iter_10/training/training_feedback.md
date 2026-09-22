# Training Feedback

## Final-policy outcome
score=53.866514, len=371.550000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[8.238277, 309.930471]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settle | 71.310000 | 30.3% | 36.1% | 57.2% |
| settle_dense | 70.073476 | 29.8% | 29.8% | 26.0% |
| terminal_success | 45.000000 | 19.1% | 19.1% | 0.0% |
| settle_potential | 1.236524 | 0.5% | 9.7% | 57.2% |
| dock_enter | 5.000000 | 2.1% | 2.1% | 0.3% |
| progress | 3.858214 | 1.6% | 1.8% | 89.6% |
| approach_cargo | 1.067534 | 0.5% | 0.9% | 99.7% |
| time_cost | -0.743100 | -0.3% | 0.3% | 100.0% |
| action_cost | -0.093349 | -0.0% | 0.0% | 100.0% |
| roughness | -0.062526 | -0.0% | 0.0% | 12.1% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
