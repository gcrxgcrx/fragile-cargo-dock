# Training Feedback

## Final-policy outcome
score=294.619915, len=244.350000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[9.194948, 310.424518]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 285.000000 | 89.3% | 89.3% | 0.4% |
| settled_bonus | 22.500000 | 7.0% | 7.0% | 4.6% |
| dock_enter | 5.000000 | 1.6% | 1.6% | 0.4% |
| progress | 4.055752 | 1.3% | 1.3% | 70.8% |
| approach_cargo | 1.219474 | 0.4% | 0.5% | 98.2% |
| roughness | -0.595012 | -0.2% | 0.2% | 24.0% |
| time_cost | -0.488700 | -0.2% | 0.2% | 100.0% |
| action_cost | -0.045414 | -0.0% | 0.0% | 100.0% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
