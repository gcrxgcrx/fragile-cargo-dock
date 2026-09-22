# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observation after action
    x_err = next_obs[0]   # horizontal position error relative to pad center
    y_err = next_obs[1]   # vertical position error (height above pad)
    vx = next_obs[2]      # horizontal velocity
    vy = next_obs[3]      # vertical velocity
    angle = next_obs[4]   # body tilt
    ang_vel = next_obs[5] # angular velocity
    left_contact = next_obs[6]  # left leg touch (0 or 1)
    right_contact = next_obs[7] # right leg touch (0 or 1)

    # ---------- Role 1: proximity_to_target (main learning signal) ----------
    dist = (x_err**2 + y_err**2)**0.5
    # Bounded negative distance to avoid extreme negative values at long range
    proximity_reward = -2.0 * (dist / (1.0 + dist))

    # ---------- Role 2: soft_landing_dynamics (stability constraint) ----------
    # Gate: activate when vertical error is small (close to landing zone)
    landing_threshold = 1.0
    gate = max(0.0, 1.0 - abs(y_err) / landing_threshold)
    # Quadratic penalty on velocities, tilt and angular velocity
    dynamics_cost = vx**2 + vy**2 + angle**2 + ang_vel**2
    soft_landing_penalty = -0.5 * gate * dynamics_cost

    # ---------- Role 3: safe_contact_encouragement (soft task completion proxy) ----------
    both_legs_contact = left_contact * right_contact  # 1 if both legs touch
    # Tolerance windows for safe landing conditions
    x_tol = 0.3
    y_tol = 0.3
    v_tol = 0.2
    angle_tol = 0.1
    angvel_tol = 0.1
    # Continuous bounded factors (1 = perfect, 0 = outside tolerance)
    factor_x = max(0.0, 1.0 - abs(x_err) / x_tol)
    factor_y = max(0.0, 1.0 - abs(y_err) / y_tol)
    factor_vx = max(0.0, 1.0 - abs(vx) / v_tol)
    factor_vy = max(0.0, 1.0 - abs(vy) / v_tol)
    factor_angle = max(0.0, 1.0 - abs(angle) / angle_tol)
    factor_angvel = max(0.0, 1.0 - abs(ang_vel) / angvel_tol)
    # Mean of factors to avoid product collapse and maintain non‑zero gradient
    mean_safety = (factor_x + factor_y + factor_vx + factor_vy + factor_angle + factor_angvel) / 6.0
    safe_contact_bonus = both_legs_contact * mean_safety * 2.0

    total_reward = proximity_reward + soft_landing_penalty + safe_contact_bonus

    components = {
        "proximity_reward": proximity_reward,
        "soft_landing_penalty": soft_landing_penalty,
        "safe_contact_bonus": safe_contact_bonus
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像与角色选择

- **task_family**: navigation_goal_reaching  
- **dynamics_subtype**: goal_approach_and_soft_contact  

根据 environment_card 中 `reward_role_decomposition` 的 mandatory roles，v1 必须包含三个职责：
1. proximity_to_target（主学习信号）
2. soft_landing_dynamics（稳定/安全约束）
3. safe_contact_encouragement（任务完成近似信号）

这些角色均有完整的可用信号（`obs[0]` 至 `obs[7]`），符合信号可用性优先原则。

## 2. 角色‑信号‑公式映射

| 角色 | 使用的信号 | 公式算子 | 形态与理由 |
|---|---|---|---|
| proximity_to_target | `x_err`, `y_err` | bounded_signal（压缩版 `-w * dist/(1+dist)`） | 减少远处极负值，保持每步连续梯度，作为核心驱动力。 |
| soft_landing_dynamics | `vx, vy, angle, ang_vel`, 高度门控用 `y_err` | quadratic_penalty + conditional_gating（线性衰减门） | 仅当垂直误差 < 1.0 时激活，避免早期压制机动。 |
| safe_contact_encouragement | `left_contact, right_contact, x_err, y_err, vx, vy, angle, ang_vel` | joint_condition_proxy（均值型） | 无显式成功 flag，用连续 bounded factor 均值代替乘积，避免稀疏塌缩。 |

## 3. 排除的角色及原因

- **fuel_efficiency**：v1 阶段暂不引入效率/动作代价，避免压制探索（留待后续迭代加入）。
- **time_to_land_bonus**：缺乏时间信号，且容易诱导危险高速行为，禁用。
- **constant_stay_alive_bonus**：与着陆目标矛盾，明确列为 avoid role。
- **angular_velocity_only_penalty (无高度门控)**：缺少门控信号，在远距离惩罚会抑制必要机动，已由 soft_landing_dynamics 中的条件惩罚覆盖。
- **terminal_success_reward / terminal_failure_penalty**：因为 `explicit_success_flag_available=false` 且 `explicit_failure_flag_available=false`，info 为空，无法使用任何终止标志。v1 用 soft proxy 代替硬成功奖励。

## 4. 为什么没有使用 terminal_success_reward 和 terminal_failure_penalty

环境没有提供 info 中的 success/failure 标志，也不允许使用未声明的终止原因。强行假设此类信号会导致奖励函数与真实终止接口脱节，违反设计原则。safe_contact_encouragement 以连续条件作为近似，引导 agent 在腿触地且状态安全时获得正向反馈，相当于构建了一个可学习的“软着陆”信号。

## 5. 留到后续迭代的职责

- **fuel_efficiency**：当 agent 具备稳定接近能力后，可加入轻量惩罚（如 `‑0.01*(action==2 or action in {1,3})`），并可能采用 training_progress 门控。
- **更精细的角速度/姿态约束**：目前仅依赖 soft_landing_dynamics 和接触奖励中的 tolerance，未来可独立组件。
- **时间效率奖励**：需要额外环境统计信息，不在纯观测奖励范围内，且容易破坏安全性，暂不引入。

## 6. 训练后应重点观察的 failure modes

- **悬停犹豫**：agent 在高空反复微调姿态，`y_err` 下降缓慢。若出现，可增强 proximity 权重或降低 soft_landing 门控的阈值，让下降信号更强。
- **高速坠毁**：末段 `y_velocity` 大负值导致 crash。可强化 soft_landing 惩罚或收紧 gate 阈值。
- **水平漂移出界**：`x_err` 持续扩大直至越界。可适当加大 proximity 权重或增加 `abs(x_err)` 的二次项。
- **侧翻/单腿接触**：双腿未同时接触且角度过大。通过 safe_contact_bonus 的 tolerance 窗口引导，若仍出现，可收紧 tolerance 或提高 contact bonus 权重。
- **过度依赖主发动机**：后期若燃料缺失，可在 v2 中引入燃料效率惩罚。