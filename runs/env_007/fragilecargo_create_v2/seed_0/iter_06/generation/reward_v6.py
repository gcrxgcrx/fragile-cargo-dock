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

    ncvx = next_obs[8] * 3.0
    ncvy = next_obs[9] * 3.0
    next_crate_speed = (ncvx * ncvx + ncvy * ncvy) ** 0.5

    # 货箱朝向对齐度：cos(heading)，1=对齐
    align_cos = obs[10]
    align_factor = (align_cos + 1.0) * 0.5  # [0,1]
    next_align_factor = (next_obs[10] + 1.0) * 0.5

    contact = obs[14]

    # ---------- 主信号 A: 势能差（货箱到坞距离改善量，bounded 压缩防极端） ----------
    raw_progress = dist - next_dist  # 靠近为正
    # 平滑压缩，避免单步极端值支配
    progress = raw_progress / (1.0 + abs(raw_progress) * 20.0)
    progress_reward = 15.0 * progress

    # ---------- 主信号 B: 完成度改善量（仅在接近坞时激活，改善量而非状态值） ----------
    # 接近门控：dist 越小越接近 1（终止边界约 0.15，阈值 0.6 给足缓冲）
    near_gate = max(0.0, 1.0 - dist / 0.6)
    # 完成度度量：进坞深度 + 对齐 + 近静止（作为"度量"用于求改善量）
    dock_measure = max(0.0, 1.0 - dist / 0.6)
    align_measure = align_factor
    slow_measure = 1.0 / (1.0 + 2.0 * crate_speed)
    completion = (dock_measure + align_measure + slow_measure) / 3.0

    next_dock_measure = max(0.0, 1.0 - next_dist / 0.6)
    next_align_measure = next_align_factor
    next_slow_measure = 1.0 / (1.0 + 2.0 * next_crate_speed)
    next_completion = (next_dock_measure + next_align_measure + next_slow_measure) / 3.0

    completion_delta = next_completion - completion  # 改善为正
    # 只在接近坞时激活，且用改善量（停留不再积累收益）
    completion_reward = 8.0 * completion_delta * near_gate

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
        + completion_reward
        + boundary_penalty
        + impact_penalty
    )

    components = {
        "crate_to_dock_progress": float(progress_reward),
        "completion_improvement": float(completion_reward),
        "boundary_avoidance": float(boundary_penalty),
        "soft_contact_penalty": float(impact_penalty),
    }

    return float(total_reward), components