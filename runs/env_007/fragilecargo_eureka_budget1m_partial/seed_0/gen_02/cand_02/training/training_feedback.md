# Training Feedback

## Final-policy outcome
score=4.524092, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[1.082143, 9.359327]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| docking_settle | 218.873860 | 76.0% | 76.0% | 53.5% |
| crate_progress_toward_dock | 56.760229 | 19.7% | 20.1% | 43.7% |
| docking_completion | 10.993258 | 3.8% | 3.8% | 2.8% |
| fragile_handling_penalty | -0.253941 | -0.1% | 0.1% | 1.7% |
| boundary_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
