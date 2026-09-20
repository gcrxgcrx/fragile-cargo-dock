# Training Feedback

## Final-policy outcome
score=51.846459, len=368.250000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[3.664772, 309.201835]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 30.000000 | 81.1% | 81.1% | 0.4% |
| crate_progress | 3.842907 | 10.4% | 11.1% | 55.6% |
| cart_approach | 0.993053 | 2.7% | 7.0% | 99.6% |
| roughness | -0.192733 | -0.5% | 0.5% | 26.9% |
| shove | -0.074028 | -0.2% | 0.2% | 2.0% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
