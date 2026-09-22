# Response Record

# 设计理由
**修改组件**：`landing_reward` 增加安全门控，避免 agent 在高速/大倾角状态下通过支撑腿接触获得大量正奖励，从而消除“冲刺→碰撞→赚奖励”的 exploit。原先公式无论落地姿态如何，只要接触就发放 `w_landing * contact_sum * landing_factor`（最大可达 12），导致 agent 为贪图高额奖励而故意高速撞击平台，引发 crash 终止。修改后：仅当瞬时合速度 ≤ 0.5 且机体倾角 ≤ 0.2 rad 时，才计算原有的 `landing_factor` 并授予正奖励；否则 `landing_factor` 置零，不给予任何鼓励。这样保留了平稳着陆的密集奖励，同时切断了 crash 行为的正反馈。其他组件（progress、velocity_penalty、orientation_penalty）保持不变，它们继续在靠近目标时抑制过快速度和过大倾角。

**系数校准**：`w_landing` 从 2.0 降至 1.0，使安全着陆单步奖励上限控制在 6.0 左右，避免主导总回报。安全阈值 `safe_speed = 0.5` 对应原 `speed_bonus` 恰好为零的边界，`safe_angle = 0.2` 约 11.5°，确保着陆足够平稳。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------- unpack observations --------------------
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # -------------------- distance to target pad --------------------
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # -------------------- derived quantities --------------------
    speed = (nvx**2 + nvy**2) ** 0.5
    abs_angle = abs(n_angle)

    # -------------------- weights --------------------
    w_progress = 2.0
    w_vel = 0.2
    w_angvel = 0.1
    w_angle = 0.15
    w_landing = 1.0          # reduced to avoid dominating
    k_speed = 2.0
    k_angle = 2.0

    # -------------------- 1. progress reward --------------------
    progress_reward = w_progress * (dist - next_dist)

    # -------------------- 2. velocity penalty (local gate) --------------------
    local_gate = max(0.0, 1.0 - next_dist)
    speed_sq = nvx**2 + nvy**2
    vel_penalty = -w_vel * speed_sq * local_gate
    ang_penalty = -w_angvel * (n_angvel**2) * local_gate

    # -------------------- 3. upright orientation incentive (gated) --------------------
    orientation_penalty = -w_angle * (abs_angle**2) * local_gate

    # -------------------- 4. safe landing reward (gated by safety) --------------------
    contact_sum = n_left_contact + n_right_contact

    # Safety gate: only reward landing if both speed and angle are within safe limits
    safe_speed = 0.5
    safe_angle = 0.2
    if speed <= safe_speed and abs_angle <= safe_angle:
        speed_bonus = max(0.0, 1.0 - k_speed * speed)
        angle_bonus = max(0.0, 1.0 - k_angle * abs_angle)
        landing_factor = 1.0 + speed_bonus + angle_bonus
    else:
        landing_factor = 0.0   # unsafe landing receives no reward

    landing_reward = w_landing * contact_sum * landing_factor

    # -------------------- total reward --------------------
    total_reward = (progress_reward +
                    vel_penalty +
                    ang_penalty +
                    orientation_penalty +
                    landing_reward)

    components = {
        "progress": progress_reward,
        "velocity_penalty": (vel_penalty + ang_penalty),
        "orientation_penalty": orientation_penalty,
        "landing_reward": landing_reward
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少 crash 前兆软信号，landing_reward 对不安全接触仍发放大量正奖励，鼓励了 exploit。
- **behavior**: agent 为获取高额 landing_reward 而以不安全姿态高速撞击平台，招致 crash 终止，导致平均 episode 长度骤降、得分崩坏。
- **signal**: landing_reward 缺少安全门控，在速度/倾角超限时仍给正反馈，纵容危险行为。
- **level**: Level 2
- **hypothesis**: 把 landing_reward 限制在速度 ≤ 0.5、倾角 ≤ 0.2 rad 的安全窗口内发放，不给 exploit 任何正向激励，迫使 agent 先减速、扶正再触地，从而稳定着陆、提升得分。
- **risk**: 安全门控使正奖励变稀疏，可能减慢初期学习速度；但已有的 velocity_penalty 和 orientation_penalty 仍提供连续梯度，应可弥补。
