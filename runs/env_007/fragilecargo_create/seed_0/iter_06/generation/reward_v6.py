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

    # 小车
    cart_fwd = obs[4]
    contact = obs[14]
    cart_x = obs[0]
    cart_y = obs[1]

    # ---------- A. MAIN: dock_approach_improvement (improvement_delta) ----------
    # 货箱到 dock 的距离（归一化坐标）
    dist_old = (dx * dx + dy * dy) ** 0.5
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    progress = dist_old - dist_new  # >0 表示货箱确实向 dock 靠近

    # 主信号：仅对"靠近"给正分，远离给负分（对称，避免刷分）
    w_progress = 25.0
    r_progress = w_progress * progress

    # ---------- B. AUX: joint_dock_completion (joint_condition_proxy, 几何平均) ----------
    # 三因子：near / slow / align，每因子连续 bounded，带 floor 防塌缩
    f_near = max(0.0, 1.0 - dist_new / 0.6)
    f_slow = 1.0 / (1.0 + 30.0 * crate_speed)
    f_align = abs(crate_cos)

    f_near_c = max(f_near, 0.05)
    f_slow_c = max(f_slow, 0.05)
    f_align_c = max(f_align, 0.05)
    joint = (f_near_c * f_slow_c * f_align_c) ** (1.0 / 3.0)

    # 辅助权重压低到主信号的 ~0.3x 量级，且不随步数无界累积
    w_joint = 0.6
    r_joint = w_joint * joint

    # ---------- C. fragile_impact_penalty (hinge, 仅接触时) ----------
    rel_speed = abs(cart_fwd - crate_speed)
    impact_excess = max(0.0, rel_speed - 0.15)
    w_impact = 4.0
    r_impact = -w_impact * impact_excess * contact

    # ---------- D. boundary_health_penalty (hinge) ----------
    bx = max(0.0, abs(cart_x) - 0.9)
    by = max(0.0, abs(cart_y) - 0.9)
    w_bounds = 6.0
    r_bounds = -w_bounds * (bx + by)

    # ---------- total ----------
    total_reward = r_progress + r_joint + r_impact + r_bounds

    components = {
        "dock_approach_improvement": r_progress,
        "joint_dock_completion": r_joint,
        "fragile_impact_penalty": r_impact,
        "boundary_health_penalty": r_bounds,
    }

    return float(total_reward), components