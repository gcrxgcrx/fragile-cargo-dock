# Training Feedback

## Final-policy outcome
score=309.639716, len=182.550000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[309.200665, 310.609361]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 300.000000 | 94.9% | 94.9% | 0.5% |
| dock_enter | 5.000000 | 1.6% | 1.6% | 0.5% |
| settle_hold | 5.000000 | 1.6% | 1.6% | 5.5% |
| progress | 4.064278 | 1.3% | 1.3% | 73.1% |
| approach_cargo | 1.150139 | 0.4% | 0.5% | 98.9% |
| time_cost | -0.365100 | -0.1% | 0.1% | 100.0% |
| roughness | -0.111634 | -0.0% | 0.0% | 27.9% |
| action_cost | -0.054918 | -0.0% | 0.0% | 100.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| out_of_bounds | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
