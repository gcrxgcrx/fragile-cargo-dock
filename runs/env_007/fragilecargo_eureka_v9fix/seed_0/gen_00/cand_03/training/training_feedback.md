# Training Feedback

## Final-policy outcome
score=8.307692, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-0.362863, 10.296015]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settle_hold | 65.000000 | 84.9% | 84.9% | 8.1% |
| dock_enter | 4.500000 | 5.9% | 5.9% | 0.2% |
| progress | 3.720480 | 4.9% | 5.3% | 85.5% |
| approach_cargo | 1.282815 | 1.7% | 2.4% | 97.6% |
| time_cost | -0.800000 | -1.0% | 1.0% | 100.0% |
| roughness | -0.162041 | -0.2% | 0.2% | 15.2% |
| action_cost | -0.153024 | -0.2% | 0.2% | 100.0% |
| boundary_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_success | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
