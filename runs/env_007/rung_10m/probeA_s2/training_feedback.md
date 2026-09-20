# Training Feedback

## Final-policy outcome
score=49.850762, len=385.950000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-0.399352, 309.583575]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 38.000000 | 86.2% | 86.2% | 0.5% |
| crate_progress | 3.601249 | 8.2% | 8.4% | 78.1% |
| cart_approach | 1.175817 | 2.7% | 4.5% | 98.4% |
| roughness | -0.308663 | -0.7% | 0.7% | 38.4% |
| shove | -0.077252 | -0.2% | 0.2% | 1.1% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
