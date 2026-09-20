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

    # 货箱朝向对齐度：cos(heading)，1=对齐
    align_cos = obs[10]
    align_factor = (align_cos + 1.0) * 0.5  # [0,1]

    contact = obs[14]

    # ---------- 主信号 A: 联合完成度状态值（几何平均，非塌缩） ----------
    # 三个成功条件各自连续 bounded factor
    # 1) 坞内位置：dist 越小越接近 1（终止边界约 0.15，用 0.6 给足梯度）
    dock_factor = max(0.0, 1.0 - dist / 0.6)
    # 2) 朝向对齐：align_factor 已在 [0,1]
    # 3) 近静止：速度越小越接近 1
    slow_factor = 1.0 / (1.0 + 3.0 * crate_speed)

    # 几何平均：任一因子为 0 时整体为 0，但比裸乘积平滑
    joint_completion = (dock_factor * align_factor * slow_factor) ** (1.0 / 3.0)

    # 凸化：强化高完成度区域的梯度，打破低水平稳态
    completion_reward = 20.0 * (joint_completion ** 2)

    # ---------- 主信号 B: 距离改善 delta（早期引导，bounded 防极端） ----------
    raw_progress = dist - next_dist  # 靠近为正
    progress = raw_progress / (1.0 + abs(raw_progress) * 20.0)
    progress_reward = 6.0 * progress

    # ---------- 组件 C: 边界安全（hinge，阈值 0.85） ----------
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
        completion_reward
        + progress_reward
        + boundary_penalty
        + impact_penalty
    )

    components = {
        "joint_completion": float(completion_reward),
        "crate_to_dock_progress": float(progress_reward),
        "boundary_avoidance": float(boundary_penalty),
        "soft_contact_penalty": float(impact_penalty),
    }

    return float(total_reward), components