# Training Feedback

## Final-policy outcome
score=-1.687025, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-2.531867, 0.183043]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_completion_state | 219.833655 | 100.0% | 100.0% | 100.0% |
| boundary_health_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_approach_improvement | 0.000000 | 0.0% | 0.0% | 0.0% |
| fragile_impact_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
