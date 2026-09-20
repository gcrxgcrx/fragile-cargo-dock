def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    dist = (dx * dx + dy * dy) ** 0.5
    next_dist = (ndx * ndx + ndy * ndy) ** 0.5

    # 货箱速率（世界系，m/s）
    crate_speed = ((obs[8] * 3.0) ** 2 + (obs[9] * 3.0) ** 2) ** 0.5

    # 货箱朝向对齐度：cos(heading)，1=对齐
    align_cos = obs[10]
    align_factor = (align_cos + 1.0) * 0.5  # [0,1]

    contact = obs[14]

    # ---------- 主信号 A: 货箱到坞的稠密推进 ----------
    # 改善量（靠近为正）
    progress = (dist - next_dist)
    # 距离凸化项：越近奖励越大，提供持续梯度（避免卡住时梯度为 0）
    dist_term = -0.5 * dist
    # 对齐门控：仅在货箱已接近坞时启用，且随接近单调增强
    align_gate = 0.0
    if dist < 0.35:
        align_gate = max(0.0, 1.0 - dist / 0.35)
    # 门控因子：对齐越好，推进奖励越高（不阻断早期探索，最低 0.3）
    gate = 0.3 + 0.7 * align_factor * align_gate
    progress_reward = 8.0 * progress * gate + 0.5 * dist_term

    # ---------- 组件 B: 入坞对齐（仅在真正接近坞时，作为门控后的辅助） ----------
    # 只在 dist < 0.15 时给少量对齐奖励，且随接近增强
    align_reward = 0.0
    if dist < 0.15:
        align_reward = 0.3 * align_factor * (1.0 - dist / 0.15)

    # ---------- 组件 C: 入坞近静止（仅在 dist < 0.1 时，hinge） ----------
    settle_penalty = 0.0
    if dist < 0.1:
        settle_gate = 1.0 - dist / 0.1
        speed_excess = max(0.0, crate_speed - 0.05)
        settle_penalty = -0.5 * speed_excess * settle_gate

    # ---------- 组件 D: 边界安全（hinge，轻量） ----------
    cart_x = obs[0]
    cart_y = obs[1]
    boundary_penalty = 0.0
    if abs(cart_x) > 0.9:
        boundary_penalty -= 0.3 * (abs(cart_x) - 0.9)
    if abs(cart_y) > 0.9:
        boundary_penalty -= 0.3 * (abs(cart_y) - 0.9)
    sensor_max = max(obs[15], obs[16], obs[17])
    if sensor_max > 0.9:
        boundary_penalty -= 0.2 * (sensor_max - 0.9)

    # ---------- 组件 E: 接触冲击抑制（仅在接触且高速时，轻罚） ----------
    impact_penalty = 0.0
    if contact > 0.5:
        impact_excess = max(0.0, crate_speed - 1.5)
        impact_penalty = -0.2 * impact_excess

    # ---------- 汇总 ----------
    total_reward = (
        progress_reward
        + align_reward
        + settle_penalty
        + boundary_penalty
        + impact_penalty
    )

    components = {
        "crate_to_dock_progress": float(progress_reward),
        "crate_dock_alignment": float(align_reward),
        "crate_settling": float(settle_penalty),
        "boundary_avoidance": float(boundary_penalty),
        "soft_contact_penalty": float(impact_penalty),
    }

    return float(total_reward), components