def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 尺度常量（依据环境卡片推导） ----
    # 货箱到 dock 距离（米）：obs[12]*5.0, obs[13]*4.0
    # 货箱速率（m/s）：obs[8]*3.0, obs[9]*3.0
    # 朝向误差：atan2(obs[11], obs[10])

    def crate_dock_dist(o):
        dx = o[12] * 5.0
        dy = o[13] * 4.0
        return (dx * dx + dy * dy) ** 0.5

    def crate_speed(o):
        vx = o[8] * 3.0
        vy = o[9] * 3.0
        return (vx * vx + vy * vy) ** 0.5

    def crate_angle_err(o):
        return abs(2.0 * (o[10] * o[10] + o[11] * o[11] + 1.0e-9) ** 0.5 - 0.0) * 0.0 + abs(
            (o[11] / ((o[10] * o[10] + o[11] * o[11]) ** 0.5 + 1.0e-9))
        )

    # 用 sin 分量近似朝向误差幅度（|sin(theta_err)|，与 |theta| 单调，小角度近似线性）
    def crate_align_err(o):
        norm = (o[10] * o[10] + o[11] * o[11]) ** 0.5
        if norm < 1.0e-6:
            return 1.0
        return abs(o[11] / norm)

    # ---- 状态量 ----
    d_prev = crate_dock_dist(obs)
    d_next = crate_dock_dist(next_obs)
    spd_prev = crate_speed(obs)
    spd_next = crate_speed(next_obs)
    align_err_next = crate_align_err(next_obs)

    contact_next = next_obs[14]

    # 接近 dock 的程度（0 远，1 在中心附近），用于门控 settling / alignment
    near_gate = 1.0 / (1.0 + 2.0 * d_next)

    # ---- 1. 主进度：货箱向 dock 靠近（delta 形式，避免悬停陷阱） ----
    progress = d_prev - d_next
    crate_to_dock_progress = 6.0 * progress

    # ---- 2. 朝向对齐（仅在接近 dock 时启用，连续 bounded） ----
    # 目标朝向误差趋近 0；|sin| 越小越好
    align_reward = near_gate * max(0.0, 1.0 - align_err_next / 0.5)
    crate_dock_alignment = 1.0 * align_reward

    # ---- 3. 停稳：接近 dock 时抑制货箱速度（hinge，仅在超阈值时罚） ----
    settle_penalty = near_gate * max(0.0, spd_next - 0.05)
    crate_settling = -3.0 * settle_penalty

    # ---- 4. 接触门控推动：接触且货箱在动时给予小幅正信号 ----
    push_signal = 0.0
    if contact_next > 0.5:
        push_signal = min(1.0, spd_next / 1.0)
    contact_gated_push = 0.5 * push_signal

    # ---- 5. 冲击柔化：接触时速度突变过大则惩罚 ----
    impact_penalty = 0.0
    if contact_next > 0.5:
        dv = abs(spd_next - spd_prev)
        impact_penalty = max(0.0, dv - 0.5)
    impact_softness = -1.0 * impact_penalty

    # ---- 6. 边界规避：小车越界风险（hinge），障碍接近度辅助 ----
    cart_x = obs[0]
    cart_y = obs[1]
    boundary_penalty = 0.0
    boundary_penalty += max(0.0, abs(cart_x) - 0.85)
    boundary_penalty += max(0.0, abs(cart_y) - 0.85)
    obstacle_closeness = max(obs[15], obs[16], obs[17])
    boundary_penalty += 0.3 * max(0.0, obstacle_closeness - 0.9)
    boundary_avoidance = -3.0 * boundary_penalty

    # ---- 7. 联合完成近似：near + align + still 的几何平均，稠密引导 ----
    f_near = 1.0 / (1.0 + 3.0 * d_next)
    f_align = max(0.0, 1.0 - align_err_next / 0.5)
    f_still = max(0.0, 1.0 - spd_next / 0.5)
    joint_proxy = (f_near * f_align * f_still) ** (1.0 / 3.0)
    joint_completion_proxy = 1.5 * joint_proxy

    components = {
        "crate_to_dock_progress": float(crate_to_dock_progress),
        "crate_dock_alignment": float(crate_dock_alignment),
        "crate_settling": float(crate_settling),
        "contact_gated_push": float(contact_gated_push),
        "impact_softness": float(impact_softness),
        "boundary_avoidance": float(boundary_avoidance),
        "joint_completion_proxy": float(joint_completion_proxy),
    }

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return (float(total_reward), components)