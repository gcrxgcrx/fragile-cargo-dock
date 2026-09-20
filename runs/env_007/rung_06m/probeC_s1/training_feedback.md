# Training Feedback

## Final-policy outcome
score=8.291958, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[7.838528, 9.186014]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_enter | 5.000000 | 42.2% | 42.2% | 0.2% |
| crate_progress | 3.227795 | 27.3% | 41.5% | 82.5% |
| cart_approach | 1.086072 | 9.2% | 14.1% | 95.2% |
| roughness | -0.169066 | -1.4% | 1.4% | 78.0% |
| shove | -0.087734 | -0.7% | 0.7% | 1.5% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_settled_hold | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_success | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
