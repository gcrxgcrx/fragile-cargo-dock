# Training Feedback

## Final-policy outcome
score=-0.123126, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-1.605258, 1.668111]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_cargo | 0.591144 | 7.5% | 49.1% | 100.0% |
| boundary_penalty | -2.899430 | -37.0% | 37.0% | 0.5% |
| time_cost | -0.800000 | -10.2% | 10.2% | 100.0% |
| progress | 0.160002 | 2.0% | 2.0% | 4.0% |
| roughness | -0.064651 | -0.8% | 0.8% | 0.4% |
| action_cost | -0.062858 | -0.8% | 0.8% | 100.0% |
| dock_enter | 0.000000 | 0.0% | 0.0% | 0.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| settle_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_success | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
