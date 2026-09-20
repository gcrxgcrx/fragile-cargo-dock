# Training Feedback

## Final-policy outcome
score=1.613820, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[0.257192, 3.155227]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_settling_and_alignment | 218.759073 | 96.6% | 96.6% | 82.4% |
| crate_to_dock_progress | 6.106604 | 2.7% | 3.3% | 42.9% |
| fragile_impact_penalty | -0.306607 | -0.1% | 0.1% | 0.2% |
| out_of_bounds_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
