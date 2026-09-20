def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    dist = (dx * dx + dy * dy) ** 0.5
    next_dist = (ndx * ndx + ndy * ndy) ** 0.5

    # 货箱世界系速率 (m/s)
    crate_speed = ((obs[8] * 3.0) ** 2 + (obs[9] * 3.0) ** 2) ** 0.5

    # 货箱朝向对齐度：cos(heading)，1=对齐
    align_cos = obs[10]
    align_factor = (align_cos + 1.0) * 0.5  # [0,1]

    contact = obs[14]

    # ---------- 主信号 A: 货箱到坞距离的改善量（唯一推进信号） ----------
    progress = dist - next_dist  # 靠近为正
    progress_reward = 12.0 * progress

    # ---------- 主信号 B: 联合完成度（进坞 + 对齐 + 近静止），仅在接近坞时激活 ----------
    # 进坞因子：dist 越小越接近 1（终止边界约 0.15，阈值 0.5 给足缓冲）
    dock_factor = max(0.0, 1.0 - dist / 0.5)
    # 对齐因子：朝向误差越小越接近 1（下限 0.15 防塌缩）
    align_gate = max(0.15, align_factor)
    # 静止因子：货箱越慢越接近 1（下限 0.15 防塌缩）
    slow_gate = max(0.15, 1.0 - crate_speed / 0.5)
    # 几何平均，避免乘积塌缩
    joint = (dock_factor * align_gate * slow_gate) ** (1.0 / 3.0)
    # 只在接近坞时激活联合完成信号，避免远处刷分
    joint_reward = 3.0 * joint * dock_factor

    # ---------- 组件 C: 边界安全（hinge，阈值 0.85 = 终止边界 1.0 的 85%） ----------
    boundary_penalty = 0.0
    if abs(obs[0]) > 0.85:
        boundary_penalty -= 0.3 * (abs(obs[0]) - 0.85)
    if abs(obs[1]) > 0.85:
        boundary_penalty -= 0.3 * (abs(obs[1]) - 0.85)
    sensor_max = max(obs[15], obs[16], obs[17])
    if sensor_max > 0.85:
        boundary_penalty -= 0.2 * (sensor_max - 0.85)

    # ---------- 组件 D: 接触冲击抑制（仅在接触且高速时，轻罚） ----------
    impact_penalty = 0.0
    if contact > 0.5:
        impact_excess = max(0.0, crate_speed - 1.5)
        impact_penalty = -0.2 * impact_excess

    # ---------- 汇总 ----------
    total_reward = (
        progress_reward
        + joint_reward
        + boundary_penalty
        + impact_penalty
    )

    components = {
        "crate_to_dock_progress": float(progress_reward),
        "joint_completion": float(joint_reward),
        "boundary_avoidance": float(boundary_penalty),
        "soft_contact_penalty": float(impact_penalty),
    }

    return float(total_reward), components