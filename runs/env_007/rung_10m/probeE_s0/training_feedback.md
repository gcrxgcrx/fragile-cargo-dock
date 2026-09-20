# Training Feedback

## Final-policy outcome
score=309.582025, len=218.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[308.997291, 310.643795]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_event | 300.000000 | 96.5% | 96.5% | 0.5% |
| dock_enter | 5.000000 | 1.6% | 1.6% | 0.5% |
| crate_progress | 4.057179 | 1.3% | 1.3% | 80.2% |
| cart_approach | 1.167748 | 0.4% | 0.5% | 98.4% |
| gentleness | -0.183818 | -0.1% | 0.1% | 22.3% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
