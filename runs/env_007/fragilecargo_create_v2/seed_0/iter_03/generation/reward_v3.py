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

    # ---------- 主信号 A: 纯推进改善量（无距离状态惩罚） ----------
    progress = dist - next_dist  # 靠近为正
    progress_reward = 8.0 * progress

    # ---------- 主信号 B: 联合完成 proxy（几何平均，防塌缩） ----------
    # f_near: 货箱接近坞（阈值 0.5，终止边界约 0.15 的 3x 缓冲）
    f_near = max(0.0, 1.0 - dist / 0.5)
    # f_align: 朝向对齐（下限 0.1 防塌缩）
    f_align = max(0.1, align_factor)
    # f_slow: 货箱近静止（速度 < 0.5 m/s 时给分，下限 0.1 防塌缩）
    f_slow = max(0.1, 1.0 - crate_speed / 0.5)
    # 几何平均
    joint_proxy = (f_near * f_align * f_slow) ** (1.0 / 3.0)
    # 仅在货箱已进入坞区附近时给联合奖励，避免早期刷分
    if dist < 0.5:
        joint_reward = 3.0 * joint_proxy
    else:
        joint_reward = 0.0

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
        "joint_completion_proxy": float(joint_reward),
        "boundary_avoidance": float(boundary_penalty),
        "soft_contact_penalty": float(impact_penalty),
    }

    return float(total_reward), components