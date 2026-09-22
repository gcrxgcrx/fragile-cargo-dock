# Training Feedback

## Final-policy outcome
score=234.452024, len=301.650000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[8.703416, 310.562494]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 225.000000 | 76.4% | 76.4% | 0.2% |
| settled_bonus | 57.300000 | 19.5% | 19.5% | 9.5% |
| dock_enter | 5.000000 | 1.7% | 1.7% | 0.3% |
| progress | 3.993429 | 1.4% | 1.4% | 80.9% |
| approach_cargo | 1.305106 | 0.4% | 0.5% | 99.2% |
| roughness | -0.740851 | -0.3% | 0.3% | 18.7% |
| time_cost | -0.603300 | -0.2% | 0.2% | 100.0% |
| action_cost | -0.072348 | -0.0% | 0.0% | 100.0% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
