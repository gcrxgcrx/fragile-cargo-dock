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

    # 货箱朝向对齐度 [0,1]：cos 误差映射
    align_cos = obs[10]
    align_factor = (align_cos + 1.0) * 0.5
    n_align_cos = next_obs[10]
    next_align_factor = (n_align_cos + 1.0) * 0.5

    contact = obs[14]

    # ---------- 联合完成度势函数 Phi ----------
    # 进入度: 距离坞越近越高, 0.5 处衰减到 0
    enter = max(0.0, 1.0 - dist / 0.5)
    next_enter = max(0.0, 1.0 - next_dist / 0.5)

    # 静止度: 速度越低越高
    still = 1.0 / (1.0 + 2.0 * crate_speed)
    next_still = 1.0 / (1.0 + 2.0 * next_crate_speed)

    # 几何平均联合 (防塌缩)
    phi = (max(enter, 1e-6) * max(align_factor, 1e-6) * max(still, 1e-6)) ** (1.0 / 3.0)
    next_phi = (max(next_enter, 1e-6) * max(next_align_factor, 1e-6) * max(next_still, 1e-6)) ** (1.0 / 3.0)

    # ---------- 主信号: 势函数改善量 ----------
    phi_delta = next_phi - phi
    progress_reward = 20.0 * phi_delta

    # 小权重状态项: 维持接近完成时的持续梯度
    completion_reward = 0.5 * next_phi

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
        "joint_completion": float(progress_reward + completion_reward),
        "crate_to_dock_progress": float(progress_reward),
        "boundary_avoidance": float(boundary_penalty),
        "soft_contact_penalty": float(impact_penalty),
    }

    return float(total_reward), components