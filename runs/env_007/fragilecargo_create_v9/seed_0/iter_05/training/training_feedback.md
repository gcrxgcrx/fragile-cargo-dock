# Training Feedback

## Final-policy outcome
score=309.545139, len=180.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[309.120308, 310.570189]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 300.000000 | 95.5% | 95.5% | 0.6% |
| dock_enter | 5.000000 | 1.6% | 1.6% | 0.6% |
| progress | 4.070160 | 1.3% | 1.3% | 78.8% |
| settle | 2.593036 | 0.8% | 0.8% | 20.4% |
| approach_cargo | 1.094736 | 0.3% | 0.6% | 99.3% |
| time_cost | -0.361800 | -0.1% | 0.1% | 100.0% |
| action_cost | -0.077889 | -0.0% | 0.0% | 100.0% |
| roughness | -0.044508 | -0.0% | 0.0% | 20.7% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
