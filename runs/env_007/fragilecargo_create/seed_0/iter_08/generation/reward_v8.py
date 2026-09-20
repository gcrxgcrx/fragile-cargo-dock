def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- unpack (仅使用环境声明的 obs 维度) ----------
    # 货箱到 dock 的有符号偏移（归一化）
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    # 货箱世界速度
    cvx = obs[8]
    cvy = obs[9]
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向
    crate_cos = obs[10]
    crate_sin = obs[11]

    # 小车前向速度、接触
    cart_fwd = obs[4]
    contact = obs[14]

    # 静态障碍接近度
    s_front = obs[15]
    s_left = obs[16]
    s_right = obs[17]

    # ---------- A. MAIN: crate_to_dock_progress (improvement_delta) ----------
    # 主信号 = 货箱到 dock 距离的逐步减少量（改善量，非状态值）
    dist_old = (dx * dx + dy * dy) ** 0.5
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    progress = dist_old - dist_new  # >0 表示货箱确实向 dock 靠近

    w_progress = 40.0
    r_progress = w_progress * progress

    # ---------- B. AUX: joint_dock_completion (低权重，几何平均，防塌缩) ----------
    # 三因子：near / slow / align，每因子连续 bounded，带 floor
    f_near = max(0.0, 1.0 - dist_new / 0.5)
    f_slow = 1.0 / (1.0 + 40.0 * crate_speed)
    f_align = abs(crate_cos)

    f_near_c = max(f_near, 0.05)
    f_slow_c = max(f_slow, 0.05)
    f_align_c = max(f_align, 0.05)
    joint = (f_near_c * f_slow_c * f_align_c) ** (1.0 / 3.0)

    # 辅助权重显著压低，避免状态占据刷分
    w_joint = 0.05
    r_joint = w_joint * joint

    # ---------- C. fragile_impact_penalty (hinge, 仅接触时) ----------
    # 相对速度超过阈值才罚，阈值设在易碎冲击边界附近
    rel_speed = abs(cart_fwd - crate_speed)
    impact_excess = max(0.0, rel_speed - 0.25)
    w_impact = 3.0
    r_impact = -w_impact * impact_excess * contact

    # ---------- D. local_obstacle_penalty (hinge, 局部障碍接近度) ----------
    obs_excess = max(0.0, s_front - 0.8) + max(0.0, s_left - 0.8) + max(0.0, s_right - 0.8)
    w_obs = 2.0
    r_obs = -w_obs * obs_excess

    # ---------- total ----------
    total_reward = r_progress + r_joint + r_impact + r_obs

    components = {
        "crate_to_dock_progress": r_progress,
        "joint_dock_completion": r_joint,
        "fragile_impact_penalty": r_impact,
        "local_obstacle_penalty": r_obs,
    }

    return float(total_reward), components