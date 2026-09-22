# Training Feedback

## Final-policy outcome
score=309.773492, len=167.850000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[309.353195, 310.735087]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 300.000000 | 95.3% | 95.3% | 0.6% |
| dock_enter | 5.000000 | 1.6% | 1.6% | 0.6% |
| progress | 4.094553 | 1.3% | 1.3% | 78.8% |
| settle | 3.621746 | 1.2% | 1.2% | 21.9% |
| approach_cargo | 1.240085 | 0.4% | 0.5% | 99.2% |
| time_cost | -0.335700 | -0.1% | 0.1% | 100.0% |
| action_cost | -0.058980 | -0.0% | 0.0% | 100.0% |
| roughness | -0.023257 | -0.0% | 0.0% | 21.5% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
