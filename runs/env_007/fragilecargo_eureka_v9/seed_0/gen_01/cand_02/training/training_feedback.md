# Training Feedback

## Final-policy outcome
score=3.529648, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[2.186783, 4.183969]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| hold_bonus | 110.350000 | 51.6% | 51.6% | 55.2% |
| success_bonus | 95.000000 | 44.5% | 44.5% | 0.2% |
| progress | 5.634899 | 2.6% | 2.7% | 40.4% |
| approach_cargo | 0.320846 | 0.2% | 0.4% | 99.7% |
| settle_shape | 0.815150 | 0.4% | 0.4% | 18.3% |
| time_cost | -0.800000 | -0.4% | 0.4% | 100.0% |
| action_cost | -0.068427 | -0.0% | 0.0% | 100.0% |
| roughness | -0.005122 | -0.0% | 0.0% | 0.0% |
| bounds_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| failure_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
