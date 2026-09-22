# Training Feedback

## Final-policy outcome
score=3.901935, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-0.924240, 8.496527]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settle_bonus | 401.900000 | 60.0% | 60.0% | 50.2% |
| terminal_success | 255.000000 | 38.1% | 38.1% | 0.2% |
| dock_enter | 4.500000 | 0.7% | 0.7% | 0.2% |
| progress | 3.649496 | 0.5% | 0.6% | 38.3% |
| approach_cargo | 0.826781 | 0.1% | 0.5% | 99.5% |
| time_cost | -0.800000 | -0.1% | 0.1% | 100.0% |
| roughness | -0.404933 | -0.1% | 0.1% | 11.8% |
| action_cost | -0.121292 | -0.0% | 0.0% | 100.0% |
| bounds_safety | 0.000000 | 0.0% | 0.0% | 0.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
