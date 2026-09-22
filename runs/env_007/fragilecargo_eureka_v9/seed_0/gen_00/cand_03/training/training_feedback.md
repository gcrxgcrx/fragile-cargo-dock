# Training Feedback

## Final-policy outcome
score=2.081349, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[1.630224, 3.304460]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 1.935401 | 20.6% | 57.1% | 74.9% |
| approach_cargo | 1.284606 | 13.7% | 30.9% | 99.2% |
| time_cost | -0.800000 | -8.5% | 8.5% | 100.0% |
| roughness | -0.225272 | -2.4% | 2.4% | 17.3% |
| action_cost | -0.103483 | -1.1% | 1.1% | 100.0% |
| bounds_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_enter | 0.000000 | 0.0% | 0.0% | 0.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_success | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
