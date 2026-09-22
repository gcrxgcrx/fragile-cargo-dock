# Training Feedback

## Final-policy outcome
score=6.672557, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[1.696811, 7.919350]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 2.668858 | 19.9% | 38.3% | 61.8% |
| dock_enter | 4.750000 | 35.3% | 35.3% | 0.2% |
| approach_cargo | 0.645150 | 4.8% | 17.2% | 99.8% |
| time_cost | -0.800000 | -6.0% | 6.0% | 100.0% |
| roughness | -0.324919 | -2.4% | 2.4% | 16.7% |
| action_cost | -0.105207 | -0.8% | 0.8% | 100.0% |
| bounds_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| settle_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_success | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
