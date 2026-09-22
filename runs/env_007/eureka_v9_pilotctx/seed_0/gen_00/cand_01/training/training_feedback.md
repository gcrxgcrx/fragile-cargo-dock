# Training Feedback

## Final-policy outcome
score=4.757237, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[0.407648, 8.785413]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 2.702284 | 29.3% | 35.6% | 77.6% |
| approach_cargo | 1.063173 | 11.5% | 25.1% | 99.1% |
| dock_enter | 2.250000 | 24.4% | 24.4% | 0.1% |
| time_cost | -0.800000 | -8.7% | 8.7% | 100.0% |
| roughness | -0.482578 | -5.2% | 5.2% | 11.5% |
| action_cost | -0.095063 | -1.0% | 1.0% | 100.0% |
| boundary_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| settled_hold | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_success | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
