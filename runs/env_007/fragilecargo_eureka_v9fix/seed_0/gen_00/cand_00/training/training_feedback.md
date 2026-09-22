# Training Feedback

## Final-policy outcome
score=23.657194, len=391.150000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[1.099534, 309.632633]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 15.000000 | 37.7% | 37.7% | 0.0% |
| settle_bonus | 12.750000 | 32.0% | 32.0% | 3.3% |
| dock_enter | 4.750000 | 11.9% | 11.9% | 0.2% |
| progress | 3.744930 | 9.4% | 10.4% | 75.3% |
| approach_cargo | 1.223981 | 3.1% | 4.0% | 99.0% |
| time_cost | -0.782300 | -2.0% | 2.0% | 100.0% |
| roughness | -0.695977 | -1.7% | 1.7% | 17.4% |
| action_cost | -0.100028 | -0.3% | 0.3% | 100.0% |
| boundary_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
