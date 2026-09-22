# Training Feedback

## Final-policy outcome
score=263.777213, len=201.950000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[1.261887, 310.199111]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 255.000000 | 95.2% | 95.2% | 0.4% |
| dock_enter | 4.250000 | 1.6% | 1.6% | 0.4% |
| progress | 3.910181 | 1.5% | 1.5% | 74.1% |
| settle | 2.396218 | 0.9% | 0.9% | 18.3% |
| approach_cargo | 1.302714 | 0.5% | 0.6% | 99.5% |
| time_cost | -0.403900 | -0.2% | 0.2% | 100.0% |
| action_cost | -0.070439 | -0.0% | 0.0% | 100.0% |
| roughness | -0.045815 | -0.0% | 0.0% | 20.9% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
