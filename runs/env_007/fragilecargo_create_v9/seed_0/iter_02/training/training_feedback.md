# Training Feedback

## Final-policy outcome
score=204.458037, len=265.600000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[8.862651, 310.006145]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 195.000000 | 92.8% | 92.8% | 0.2% |
| dock_enter | 5.000000 | 2.4% | 2.4% | 0.4% |
| progress | 4.049457 | 1.9% | 2.0% | 64.9% |
| settle | 3.386180 | 1.6% | 1.6% | 39.0% |
| approach_cargo | 1.203043 | 0.6% | 0.9% | 99.6% |
| time_cost | -0.531200 | -0.3% | 0.3% | 100.0% |
| action_cost | -0.073637 | -0.0% | 0.0% | 100.0% |
| roughness | -0.065634 | -0.0% | 0.0% | 15.6% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
