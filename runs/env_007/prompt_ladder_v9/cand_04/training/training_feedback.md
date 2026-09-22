# Training Feedback

## Final-policy outcome
score=0.732620, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[0.139234, 2.177041]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 2.013660 | 21.7% | 53.9% | 56.6% |
| approach_cargo | -0.188647 | -2.0% | 33.7% | 99.1% |
| time_cost | -0.800000 | -8.6% | 8.6% | 100.0% |
| roughness | -0.242608 | -2.6% | 2.6% | 16.8% |
| action_cost | -0.080307 | -0.9% | 0.9% | 100.0% |
| hard_hit | -0.025000 | -0.3% | 0.3% | 0.0% |
| boundary_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_enter | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_success | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
