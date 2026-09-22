# Training Feedback

## Final-policy outcome
score=0.177880, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-0.367902, 1.338296]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_cargo | 0.987077 | 45.5% | 60.5% | 100.0% |
| time_cost | -0.800000 | -36.9% | 36.9% | 100.0% |
| action_cost | -0.027849 | -1.3% | 1.3% | 100.0% |
| progress | 0.021757 | 1.0% | 1.0% | 4.7% |
| roughness | -0.006589 | -0.3% | 0.3% | 0.2% |
| bounds_safety | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_enter | 0.000000 | 0.0% | 0.0% | 0.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| settle_dwell | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_success | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
