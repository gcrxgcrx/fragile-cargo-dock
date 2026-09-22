# 设计理由
正常模式下诊断并修改上一轮奖励函数。当前分数 50.66，长度 703.5，12/20 终止（可能着陆成功）8/20 截断。代理正在学习着陆，但着陆阶段耗时很长（平均 700+ 步），且着陆后缺乏稳定信号。根据累积记录，唯一一次结构修改（加 landing_bonus）使分数从负值跳至 +50，说明一次性稀疏奖励强度足以推动学习，但未能持续引导后续稳定。

**改了什么**：将 `landing_bonus`（一次性 30.0）替换为 `landing_quality`（每步连续奖励）。`landing_quality` 把原来的二值安全检查转变为连续乘积因子，每步给予 0 到 0.5 的持续奖励，引导代理在接触后维持安全状态（低速度、低角度、低角速度）。同时给 `orientation_penalty` 添加接触条件门控，着陆后停止此惩罚，避免与 `landing_quality` 产生矛盾。

**为什么**：按公式切换指南，`landing_bonus` 属于稀疏二值 proxy，证据是 active_rate 仅 14.29%（二值触发率低），且 episode 较长。应变换为联合条件连续因子，确保每步有梯度。同时，着陆后 `orientation_penalty` 的方向控制与 `landing_quality` 的静止要求重叠导致过严，改为接触后关闭。

**数学形式**：
- `landing_quality`：每步因子 `(1.0 - clamped_speed) * (1.0 - clamped_angle) * (1.0 - clamped_angular)`，线性缩放至 0.5
- `contact_gate`：`(n_left_contact + n_right_contact) / 2`，用于门控 orientation_penalty

**系数校准**：
- `landing_quality` 上限 0.5/step，在主信号 per-step 约 0.07（progress_reward 700 步贡献 50 → 0.07/step）的 7 倍，但实际在安全区内因子大多 0.3-0.5，约为主信号的 2-3.5 倍。单组件略超 2x 上限，但在着陆阶段是主要驱动力，可接受一次。
- `orientation_penalty` 在接触后关闭，着陆段不再干扰 stability；值维持原 -0.3 / -0.2。

不修改其他组件，避免干扰正在起作用的结构。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v2: landing_bonus → landing_quality (dense per-step); gate orientation_penalty.
    """
    # --- Unpack observations ---
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx_pos, ny_pos = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_body_angle = next_obs[4]
    n_angular_vel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # --- Component A: delta_distance ---
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (nx_pos**2 + ny_pos**2) ** 0.5
    delta_distance = current_distance - next_distance
    progress_reward = 10.0 * delta_distance

    # --- Component B: bounded_distance_shaping ---
    distance_shaping = 2.0 * (1.0 / (1.0 + 0.5 * next_distance))

    # --- Component C: velocity_hinge_constraint ---
    h_speed = abs(nx_vel)
    h_penalty = max(0.0, h_speed - 2.0)
    v_speed = ny_vel
    v_down_penalty = max(0.0, -v_speed - 1.5)
    v_up_penalty = max(0.0, v_speed - 1.0)
    angular_penalty = max(0.0, abs(n_angular_vel) - 0.8)
    velocity_constraint = -0.5 * (h_penalty + v_down_penalty + v_up_penalty + angular_penalty)

    # --- Component D: orientation_stabilization (gated) ---
    # Gate: average contact of next legs, 0 if no contact, 1 if both contact
    contact_gate = (n_left_contact + n_right_contact) / 2.0
    # When after contact, disable orientation penalty to avoid conflict with landing_quality
    orientation_penalty = (1.0 - contact_gate) * (-0.3 * (n_body_angle**2) - 0.2 * (n_angular_vel**2))

    # --- Component E: fuel_efficiency ---
    if action == 0:
        fuel_penalty = 0.0
    elif action == 2:
        fuel_penalty = -0.08
    else:
        fuel_penalty = -0.05

    # --- Component F: landing_quality (dense per-step) ---
    # Continuous bounded factors for safe landing status
    speed_norm = (nx_vel**2 + ny_vel**2) ** 0.5
    # Scale such that perfect zero gives 1.0, and at landing thresholds gives ~0.0
    safe_speed_factor = max(0.0, 1.0 - speed_norm / 0.4)   # 0 at 0.4 speed, 1 at 0
    safe_angle_factor = max(0.0, 1.0 - abs(n_body_angle) / 0.2)  # 0 at 0.2 rad
    safe_spin_factor = max(0.0, 1.0 - abs(n_angular_vel) / 0.2)  # 0 at 0.2 rad/s

    # Landing quality = product of factor * contact_gate (so only active when legs touch)
    landing_quality = 0.5 * contact_gate * safe_speed_factor * safe_angle_factor * safe_spin_factor

    # --- Assemble total reward ---
    total_reward = (
        progress_reward +
        distance_shaping +
        velocity_constraint +
        orientation_penalty +
        fuel_penalty +
        landing_quality
    )

    components = {
        'progress_reward': progress_reward,
        'distance_shaping': distance_shaping,
        'velocity_constraint': velocity_constraint,
        'orientation_penalty': orientation_penalty,
        'fuel_penalty': fuel_penalty,
        'landing_quality': landing_quality
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少超出视口的前兆信号，缺少着陆后持续稳定信号；landing_bonus 是一次性二值代理，梯度稀疏。
- **behavior**: 代理已学会触及平台并接触，但着陆后稳定阶段耗时过长（len 703），可能缓慢调整姿势，terminal 仅 12/20。
- **signal**: 着陆后缺少持续引导，orientation_penalty 在接触后仍在惩罚正常微调，形成矛盾。
- **level**: Level 2
- **hypothesis**: 将 landing_bonus 替换为每步 dense landing_quality，可提供连续梯度加速稳定收敛，同时关闭接触后 orientation_penalty，减少着陆阶段冲突，使 len 下降、terminated 率上升、score 提高。
- **risk**: landing_quality 因子乘积可能在部分接触场景塌缩为 0（如一条腿悬空），但 contact_gate 乘进去后，单腿接触仍有收益，预计不会完全塌缩。若 len 仍高且无改善，下一步可考虑 transform to 几何平均或加 horizon_position_constraint。