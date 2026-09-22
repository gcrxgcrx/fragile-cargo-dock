# Training Feedback

## Final-policy outcome
score=-1.778906, len=394.800000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-97.367039, 3.931164]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_bonus | 30.062580 | 47.1% | 47.1% | 56.3% |
| dock_hold_bonus | 12.000000 | 18.8% | 18.8% | 0.2% |
| dwell | 11.984248 | 18.8% | 18.8% | 6.1% |
| progress | 5.375478 | 8.4% | 9.4% | 49.8% |
| approach_cargo | 0.477724 | 0.7% | 2.2% | 99.7% |
| safety_guard | -1.250000 | -2.0% | 2.0% | 0.0% |
| time_cost | -0.394800 | -0.6% | 0.6% | 100.0% |
| bounds_penalty | -0.358958 | -0.6% | 0.6% | 1.3% |
| roughness | -0.223226 | -0.3% | 0.3% | 2.2% |
| action_cost | -0.071394 | -0.1% | 0.1% | 100.0% |
| hard_hit | -0.050000 | -0.1% | 0.1% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
