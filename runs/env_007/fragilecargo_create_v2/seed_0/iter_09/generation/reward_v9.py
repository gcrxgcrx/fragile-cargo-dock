def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    dist = (dx * dx + dy * dy) ** 0.5
    next_dist = (ndx * ndx + ndy * ndy) ** 0.5

    # 货箱世界系速率 (m/s)
    cvx = obs[8] * 3.0
    cvy = obs[9] * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向对齐度 [0,1]
    align_cos = obs[10]
    align_factor = (align_cos + 1.0) * 0.5

    contact = obs[14]

    # ---------- 主信号: 推进改善量 (bounded) ----------
    raw_progress = dist - next_dist  # 靠近为正
    progress = raw_progress / (1.0 + abs(raw_progress) * 20.0)

    # ---------- 联合门控: 仅在接近坞时激活静止/对齐因子 ----------
    dock_factor = max(0.0, 1.0 - dist / 0.6)  # 距离坞 <0.6 才 >0

    # 关键: slow_factor 只在已接近坞时生效, 否则=1 (杜绝远处静止收分)
    if dock_factor > 0.3:
        slow_factor = 1.0 / (1.0 + 3.0 * crate_speed)
    else:
        slow_factor = 1.0

    # 几何平均联合门控 (避免乘积塌缩)
    gate = (max(dock_factor, 1e-6) * max(align_factor, 1e-6) * max(slow_factor, 1e-6)) ** (1.0 / 3.0)

    # 推进收益: 基础 + 门控放大 (接近目标时推进更值钱)
    progress_reward = 10.0 * progress * (1.0 + 2.0 * gate)

    # ---------- 完成度状态值 (极小权重, 且必须乘 dock_factor) ----------
    completion_reward = 0.15 * dock_factor * (gate ** 2)

    # ---------- 边界安全 (hinge, 阈值 0.85) ----------
    boundary_penalty = 0.0
    if abs(obs[0]) > 0.85:
        boundary_penalty -= 0.3 * (abs(obs[0]) - 0.85)
    if abs(obs[1]) > 0.85:
        boundary_penalty -= 0.3 * (abs(obs[1]) - 0.85)
    sensor_max = max(obs[15], obs[16], obs[17])
    if sensor_max > 0.85:
        boundary_penalty -= 0.2 * (sensor_max - 0.85)

    # ---------- 接触冲击抑制 (仅接触且高速时) ----------
    impact_penalty = 0.0
    if contact > 0.5:
        impact_excess = max(0.0, crate_speed - 1.5)
        impact_penalty = -0.2 * impact_excess

    # ---------- 汇总 ----------
    total_reward = (
        progress_reward
        + completion_reward
        + boundary_penalty
        + impact_penalty
    )

    components = {
        "crate_to_dock_progress": float(progress_reward),
        "joint_completion_gate": float(completion_reward),
        "boundary_avoidance": float(boundary_penalty),
        "soft_contact_penalty": float(impact_penalty),
    }

    return float(total_reward), components