# EUREKA-style population search - summary

- environment: `FragileCargoDock-v0`
- generations: 1, population 2 (elite 1 x 1 + 1 children each)
- timesteps per candidate: 1,200,000
- evaluation episodes: 20
- selection metric: native `external_eval.mean_eval_reward`
- seed: 0

**Best candidate: `g00c00` with native score 172.6730**

## Generation 0

| id | parent | native score |
|---|---|---:|
| g00c00 | - | 172.6730 |
| g00c01 | - | 0.0360 |

## Best reward

`runs\env_007\eureka_v9_fixedspec\seed_0\gen_00\cand_00\reward_v1.py`

copied to `runs\env_007\eureka_v9_fixedspec\seed_0\best\best_reward.py`