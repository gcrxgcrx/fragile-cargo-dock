# Training Feedback

## Final-policy outcome
score=294.207092, len=182.500000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-0.603366, 310.497171]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 285.000000 | 95.5% | 95.5% | 0.5% |
| dock_enter | 4.750000 | 1.6% | 1.6% | 0.5% |
| progress | 3.893926 | 1.3% | 1.3% | 74.6% |
| settle | 2.539041 | 0.9% | 0.9% | 19.5% |
| approach_cargo | 1.245577 | 0.4% | 0.5% | 98.6% |
| time_cost | -0.365000 | -0.1% | 0.1% | 100.0% |
| action_cost | -0.065359 | -0.0% | 0.0% | 100.0% |
| roughness | -0.038747 | -0.0% | 0.0% | 25.7% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
