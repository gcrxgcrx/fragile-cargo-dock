# Training Feedback

## Final-policy outcome
score=0.036024, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-1.063135, 1.924584]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_cargo | 0.492546 | 16.4% | 54.0% | 100.0% |
| time_cost | -0.800000 | -26.6% | 26.6% | 100.0% |
| progress | 0.463207 | 15.4% | 15.5% | 20.1% |
| roughness | -0.080647 | -2.7% | 2.7% | 2.2% |
| action_cost | -0.037428 | -1.2% | 1.2% | 100.0% |
| boundary_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_enter | 0.000000 | 0.0% | 0.0% | 0.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| settle_step | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_success | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
