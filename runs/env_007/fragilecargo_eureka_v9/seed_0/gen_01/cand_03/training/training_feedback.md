# Training Feedback

## Final-policy outcome
score=3.609777, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[3.098483, 4.697044]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settle_bonus | 81.190008 | 83.5% | 83.5% | 63.5% |
| progress | 7.672441 | 7.9% | 7.9% | 47.8% |
| dock_enter | 5.000000 | 5.1% | 5.1% | 0.2% |
| approach_cargo | 0.426802 | 0.4% | 1.3% | 99.8% |
| roughness | -1.162910 | -1.2% | 1.2% | 9.9% |
| time_cost | -0.800000 | -0.8% | 0.8% | 100.0% |
| action_cost | -0.093564 | -0.1% | 0.1% | 100.0% |
| bounds_safety | 0.000000 | 0.0% | 0.0% | 0.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_success | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
