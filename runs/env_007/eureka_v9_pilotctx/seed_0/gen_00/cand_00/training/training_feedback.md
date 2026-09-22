# Training Feedback

## Final-policy outcome
score=203.796423, len=287.600000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[0.351667, 310.148780]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 195.000000 | 93.1% | 93.1% | 0.2% |
| dock_enter | 4.500000 | 2.1% | 2.1% | 0.3% |
| progress | 3.817017 | 1.8% | 1.9% | 79.4% |
| settled_bonus | 3.750000 | 1.8% | 1.8% | 2.6% |
| approach_cargo | 1.307897 | 0.6% | 0.7% | 95.4% |
| time_cost | -0.575200 | -0.3% | 0.3% | 100.0% |
| action_cost | -0.058882 | -0.0% | 0.0% | 100.0% |
| roughness | -0.038034 | -0.0% | 0.0% | 27.8% |
| boundary_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
