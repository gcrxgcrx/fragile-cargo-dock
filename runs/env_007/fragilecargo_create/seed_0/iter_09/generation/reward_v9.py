def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- unpack (仅使用环境声明的 obs 维度) ----------
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    cvx = obs[8]
    cvy = obs[9]
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    crate_cos = obs[10]
    crate_sin = obs[11]

    cart_fwd = obs[4]
    contact = obs[14]

    # ---------- A. MAIN: dock_completion_joint (几何平均, 连续 bounded) ----------
    # 三因子：near / slow / align，每因子连续 bounded，带 floor 防塌缩
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    f_near = max(0.0, 1.0 - dist_new / 0.5)
    f_slow = 1.0 / (1.0 + 40.0 * crate_speed)
    f_align = abs(crate_cos)

    f_near_c = max(f_near, 0.05)
    f_slow_c = max(f_slow, 0.05)
    f_align_c = max(f_align, 0.05)
    joint = (f_near_c * f_slow_c * f_align_c) ** (1.0 / 3.0)

    w_joint = 30.0
    r_joint = w_joint * joint

    # ---------- B. AUX: crate_to_dock_progress (改善量, 低权重) ----------
    dist_old = (dx * dx + dy * dy) ** 0.5
    progress = dist_old - dist_new
    w_progress = 8.0
    r_progress = w_progress * progress

    # ---------- C. crate_speed_hinge (停稳引导, 只在货箱过快时罚) ----------
    # 阈值设在成功速度边界(0.05)之上，给安全区留空间
    speed_excess = max(0.0, crate_speed - 0.15)
    w_speed = 6.0
    r_speed = -w_speed * speed_excess

    # ---------- D. fragile_impact_penalty (hinge, 仅接触时) ----------
    rel_speed = abs(cart_fwd - crate_speed)
    impact_excess = max(0.0, rel_speed - 0.25)
    w_impact = 3.0
    r_impact = -w_impact * impact_excess * contact

    # ---------- total ----------
    total_reward = r_joint + r_progress + r_speed + r_impact

    components = {
        "dock_completion_joint": r_joint,
        "crate_to_dock_progress": r_progress,
        "crate_speed_hinge": r_speed,
        "fragile_impact_penalty": r_impact,
    }

    return float(total_reward), components