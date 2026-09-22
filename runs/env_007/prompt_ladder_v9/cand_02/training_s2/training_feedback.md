# Training Feedback

## Final-policy outcome
score=1.969646, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-0.423097, 2.860134]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_cargo | 0.907614 | 15.3% | 43.3% | 93.8% |
| progress | 2.123225 | 35.9% | 36.0% | 46.6% |
| time_cost | -0.800000 | -13.5% | 13.5% | 100.0% |
| roughness | -0.334047 | -5.6% | 5.6% | 9.0% |
| action_cost | -0.093948 | -1.6% | 1.6% | 100.0% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_enter | 0.000000 | 0.0% | 0.0% | 0.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| settled_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_success | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
