# Training Feedback

## Final-policy outcome
score=0.283800, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-0.175837, 1.275866]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| out_of_bounds | -31.165007 | -92.4% | 92.4% | 3.1% |
| action_cost | -1.793744 | -5.3% | 5.3% | 100.0% |
| gentleness | -0.708069 | -2.1% | 2.1% | 0.9% |
| crate_progress | 0.051010 | 0.2% | 0.2% | 5.9% |
| enter_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |
| settle_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
