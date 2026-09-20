# Training Feedback

## Final-policy outcome
score=264.058490, len=279.200000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[2.852395, 309.813153]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_event | 255.000000 | 96.0% | 96.0% | 0.3% |
| dock_enter | 4.750000 | 1.8% | 1.8% | 0.3% |
| crate_progress | 3.905883 | 1.5% | 1.5% | 82.2% |
| cart_approach | 1.221972 | 0.5% | 0.6% | 99.6% |
| gentleness | -0.193191 | -0.1% | 0.1% | 26.0% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
