# Training Feedback

## Final-policy outcome
score=156.686499, len=349.050000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[2.901552, 310.342269]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 150.000000 | 87.2% | 87.2% | 0.1% |
| settled_hold | 12.400000 | 7.2% | 7.2% | 1.8% |
| progress | 3.873180 | 2.3% | 2.3% | 81.8% |
| dock_enter | 2.500000 | 1.5% | 1.5% | 0.1% |
| approach_cargo | 1.255487 | 0.7% | 1.0% | 99.2% |
| time_cost | -0.698100 | -0.4% | 0.4% | 100.0% |
| roughness | -0.575377 | -0.3% | 0.3% | 17.9% |
| action_cost | -0.076171 | -0.0% | 0.0% | 100.0% |
| boundary_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
