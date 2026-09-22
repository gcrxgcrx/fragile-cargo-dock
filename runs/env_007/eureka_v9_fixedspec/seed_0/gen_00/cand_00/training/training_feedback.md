# Training Feedback

## Final-policy outcome
score=172.672976, len=348.150000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[3.137113, 309.670031]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 165.000000 | 90.6% | 90.6% | 0.2% |
| settle_step | 5.500000 | 3.0% | 3.0% | 1.6% |
| progress | 3.885012 | 2.1% | 2.2% | 83.1% |
| dock_enter | 3.750000 | 2.1% | 2.1% | 0.2% |
| approach_cargo | 1.009255 | 0.6% | 1.2% | 98.7% |
| roughness | -0.919682 | -0.5% | 0.5% | 15.0% |
| time_cost | -0.696300 | -0.4% | 0.4% | 100.0% |
| action_cost | -0.061959 | -0.0% | 0.0% | 100.0% |
| boundary_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
