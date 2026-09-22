# Training Feedback

## Final-policy outcome
score=36.094063, len=377.200000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-0.306291, 309.776369]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 30.000000 | 73.2% | 73.2% | 0.0% |
| progress | 3.118904 | 7.6% | 8.8% | 49.9% |
| dock_enter | 3.250000 | 7.9% | 7.9% | 0.2% |
| approach_cargo | 0.884230 | 2.2% | 5.0% | 96.0% |
| settled_hold | 1.000000 | 2.4% | 2.4% | 0.3% |
| time_cost | -0.754400 | -1.8% | 1.8% | 100.0% |
| roughness | -0.224643 | -0.5% | 0.5% | 15.0% |
| action_cost | -0.077102 | -0.2% | 0.2% | 100.0% |
| hard_hit | -0.025000 | -0.1% | 0.1% | 0.0% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
