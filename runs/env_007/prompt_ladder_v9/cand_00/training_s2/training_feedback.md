# Training Feedback

## Final-policy outcome
score=309.636551, len=264.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[309.233443, 310.525712]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 300.000000 | 92.0% | 92.0% | 0.4% |
| settle_hold | 14.575000 | 4.5% | 4.5% | 11.0% |
| dock_enter | 5.000000 | 1.5% | 1.5% | 0.4% |
| progress | 4.071959 | 1.2% | 1.3% | 85.4% |
| approach_cargo | 1.298897 | 0.4% | 0.5% | 97.5% |
| time_cost | -0.528800 | -0.2% | 0.2% | 100.0% |
| roughness | -0.360892 | -0.1% | 0.1% | 30.7% |
| action_cost | -0.053975 | -0.0% | 0.0% | 100.0% |
| hard_hit | -0.025000 | -0.0% | 0.0% | 0.0% |
| out_of_bounds | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
