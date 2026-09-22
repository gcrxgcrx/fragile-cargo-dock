# Training Feedback

## Final-policy outcome
score=-2.107220, len=395.350000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-100.093385, 8.052156]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| docked_bonus | 601.910302 | 64.9% | 64.9% | 53.8% |
| settle_bonus | 191.200000 | 20.6% | 20.6% | 48.4% |
| terminal_success | 120.000000 | 12.9% | 12.9% | 0.2% |
| progress | 5.263858 | 0.6% | 0.7% | 45.0% |
| terminal_failure | -3.000000 | -0.3% | 0.3% | 0.0% |
| approach_cargo | 0.093290 | 0.0% | 0.3% | 99.7% |
| time_cost | -0.790700 | -0.1% | 0.1% | 100.0% |
| bounds_penalty | -0.703083 | -0.1% | 0.1% | 0.5% |
| roughness | -0.265852 | -0.0% | 0.0% | 11.3% |
| action_cost | -0.105766 | -0.0% | 0.0% | 100.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
