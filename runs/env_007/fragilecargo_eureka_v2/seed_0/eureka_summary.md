# EUREKA-style population search - summary

- environment: `FragileCargoDock-v0`
- generations: 4, population 4 (elite 2 x 1 + 1 children each)
- timesteps per candidate: 3,000,000
- evaluation episodes: 20
- selection metric: native `external_eval.mean_eval_reward`
- seed: 0

**Best candidate: `g02c03` with native score 8.0927**

## Generation 0

| id | parent | native score |
|---|---|---:|
| g00c02 | - | 3.6796 |
| g00c03 | - | 3.0174 |
| g00c00 | - | 2.3414 |
| g00c01 | - | -7.0554 |

## Generation 1

| id | parent | native score |
|---|---|---:|
| g01c02 | g00c02 | 3.7051 |
| g01e00 | g00c02 | 3.6796 |
| g01e01 | g00c03 | 3.0174 |
| g01c03 | g00c03 | 2.9883 |

## Generation 2

| id | parent | native score |
|---|---|---:|
| g02c03 | g01e00 | 8.0927 |
| g02c02 | g01c02 | 3.7488 |
| g02e00 | g01c02 | 3.7051 |
| g02e01 | g01e00 | 3.6796 |

## Generation 3

| id | parent | native score |
|---|---|---:|
| g03e00 | g02c03 | 8.0927 |
| g03e01 | g02c02 | 3.7488 |
| g03c02 | g02c03 | 2.4497 |
| g03c03 | g02c02 | 1.4543 |

## Best reward

`runs\env_007\fragilecargo_eureka_v2\seed_0\gen_02\cand_03\reward_v1.py`

copied to `runs\env_007\fragilecargo_eureka_v2\seed_0\best\best_reward.py`