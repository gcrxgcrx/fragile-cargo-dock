# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations (origin at target platform center/height)
    x_pos, y_pos, x_vel, y_vel, body_angle, ang_vel, l_contact, r_contact = obs
    nx_pos, ny_pos, nx_vel, ny_vel, n_angle, n_ang_vel, nl_contact, nr_contact = next_obs

    # Distance to target
    current_dist = (x_pos**2 + y_pos**2) ** 0.5
    next_dist    = (nx_pos**2 + ny_pos**2) ** 0.5

    # Speed magnitude
    current_speed = (x_vel**2 + y_vel**2) ** 0.5
    next_speed    = (nx_vel**2 + ny_vel**2) ** 0.5

    # ---------- Component A: potential‑based shaping for approach + deceleration ----------
    w_dist = 1.0
    w_speed = 0.5
    phi_current = -(w_dist * current_dist + w_speed * current_speed)
    phi_next    = -(w_dist * next_dist    + w_speed * next_speed)
    progress = phi_next - phi_current   # positive when distance shrinks or speed reduces

    # ---------- Component B: upright orientation penalty ----------
    orientation_penalty = -0.1 * (n_angle ** 2) - 0.05 * (n_ang_vel ** 2)

    # ---------- Component C: fuel cost ----------
    if action == 0:                     # no engine
        fuel_cost = 0.0
    elif action == 2:                   # main engine (most costly)
        fuel_cost = -0.1
    else:                               # orientation engines (1 or 3)
        fuel_cost = -0.05

    # ---------- Total ----------
    total = progress + orientation_penalty + fuel_cost

    components = {
        "progress": progress,
        "orientation_penalty": orientation_penalty,
        "fuel_cost": fuel_cost
    }
    return float(total), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- `task_family`: navigation_goal_reaching  
- `dynamics_subtype`: goal_approach_and_soft_contact (2D 着陆器，离散引擎)

## selected reward roles
按 `reward_role_decomposition` 的 mandatory 职责，v1 选取了四个角色：
1. **distance_to_target（主学习信号）** – 结合 soft_landing_velocity 通过 potential‑based shaping 实现。
2. **soft_landing_velocity** – 与距离一起纳入同一个 potential，鼓励同时减距离和减速度。
3. **upright_orientation** – 对小车身角度和角速度进行二次惩罚。
4. **fuel_penalty** – 对发动机动作施加轻量固定代价。

## role‑to‑signal mapping
| role | signals used | formula operator |
|---|---|---|
| combined approach + deceleration | `next_obs[0], next_obs[1]` → distance<br>`next_obs[2], next_obs[3]` → speed | `potential_based_shaping` (γ=1) : `w_dist*(dist – next_dist) + w_speed*(speed – next_speed)` |
| upright_orientation | `next_obs[4]` (body_angle)<br>`next_obs[5]` (angular_velocity) | `dense_state_signal` (quadratic penalty) |
| fuel_penalty | `action` (0~3) | constant per non‑zero action; higher for main engine (action=2) |

## excluded roles & why
- **landing_contact_bonus** – 双腿接触是着陆结果，且信号稀疏；v1 先通过 potential 让 agent 自然减速并接近，再考虑后续加入 contact‑conditioned reward。
- **terminal_success_reward / terminal_failure_penalty** – `info` 为空，缺少显式成功/失败标志。
- **time_penalty** – 无可靠步数/时间信号，且 training_progress 未授权使用。
- **crash_avoidance_penalty** – 当前通过速度 penalty 已间接防止高速坠毁，复杂 crash 检测留到后续。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
环境卡片声明 `explicit_success_flag_available: false`，`info` 为空字典，函数参数列表中没有 `done`，因此既不能判断 episode 是否结束，也无法知道终止原因。一切奖励必须基于逐步提供的观测构建。

## 哪些职责留到后续迭代
- landing_contact_bonus（带速度、姿态条件的连续乘积奖励）
- crash_avoidance_penalty（精细的低空高速检测）
- 动作平滑性惩罚（需要历史动作，当前接口不提供）
- 效率/能耗的精细优化（例如基于持续时间的动态权重）

## 训练后应重点观察的 failure modes
1. **减速不足**：agent 到达平台时仍保有较高速度，导致穿透或弹跳。需要增大 `w_speed` 或在靠近平台时加入条件速度 hinge。
2. **过度谨慎**：燃料惩罚过重导致 agent 全程不敢点火，长时间漂浮而不下降。需降低燃料代价或引入时间压力。
3. **姿态振荡**：转向发动机频繁使用但未稳定姿态，角速度惩罚不足。可微调 orientation 系数。
4. **reward 尺度失衡**：progress 的量级远大于 penalty，使 agent 忽略姿态而追求快速降落，增加 crash 风险。需监控各组件 episode_sum_mean。
