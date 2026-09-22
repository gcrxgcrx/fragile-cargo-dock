# Training Feedback

## Final-policy outcome
score=309.735200, len=203.600000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[309.340665, 310.606473]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 300.000000 | 93.9% | 93.9% | 0.5% |
| settle_hold | 8.050000 | 2.5% | 2.5% | 7.9% |
| dock_enter | 5.000000 | 1.6% | 1.6% | 0.5% |
| progress | 4.055667 | 1.3% | 1.3% | 78.2% |
| approach_cargo | 1.305317 | 0.4% | 0.5% | 99.4% |
| time_cost | -0.407200 | -0.1% | 0.1% | 100.0% |
| roughness | -0.164378 | -0.1% | 0.1% | 27.7% |
| action_cost | -0.065196 | -0.0% | 0.0% | 100.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| out_of_bounds | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
