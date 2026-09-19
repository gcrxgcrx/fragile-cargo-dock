def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 信号提取（仅使用声明的 obs 维度） ----
    cur_dx = obs[12] * 5.0
    cur_dy = obs[13] * 4.0
    nxt_dx = next_obs[12] * 5.0
    nxt_dy = next_obs[13] * 4.0

    cur_dist = (cur_dx * cur_dx + cur_dy * cur_dy) ** 0.5
    nxt_dist = (nxt_dx * nxt_dx + nxt_dy * nxt_dy) ** 0.5

    # 货箱速率 (m/s)
    cur_crate_speed = ((obs[8] * 3.0) ** 2 + (obs[9] * 3.0) ** 2) ** 0.5
    nxt_crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5

    # 货箱朝向对齐程度：|cos(theta)| 越接近 1 越对齐
    crate_heading = (next_obs[11] * next_obs[11] + next_obs[10] * next_obs[10]) ** 0.5
    if crate_heading < 1e-6:
        align_cos = 0.0
    else:
        align_cos = abs(next_obs[10] / crate_heading)

    # 接触标志
    contact = next_obs[14]

    # 小车速度/角速度
    cart_speed = next_obs[4] * 3.0
    cart_yaw = next_obs[5] * 8.0

    # 时间比例
    time_frac = next_obs[18]
    if time_frac < 0.0:
        time_frac = 0.0
    if time_frac > 1.0:
        time_frac = 1.0

    # 障碍接近度
    front_prox = next_obs[15]

    # ---- 组件 1: crate_to_dock_progress ----
    # 势能差（距离减少量）保留，作为稠密方向信号
    delta_dist = cur_dist - nxt_dist
    if delta_dist > 0.5:
        delta_dist = 0.5
    if delta_dist < -0.5:
        delta_dist = -0.5
    progress_reward = 6.0 * delta_dist

    # ---- 组件 2: dock_proximity（凸化稠密信号，替代悬停刷分） ----
    # 用 1/(1+d)^2 形式，距离越近梯度越强，且远离 dock 时接近 0
    # 关键：这是"进入 dock"的主要梯度来源
    dock_prox = 1.0 / (1.0 + 1.5 * nxt_dist)
    dock_prox = dock_prox * dock_prox
    proximity_reward = 4.0 * dock_prox

    # ---- 组件 3: contact_gated_push ----
    # 接触时，货箱速度朝向 dock 的分量越大越好
    if contact > 0.5 and nxt_dist > 0.2:
        if cur_dist > 1e-6:
            dir_x = -cur_dx / cur_dist
            dir_y = -cur_dy / cur_dist
        else:
            dir_x = 0.0
            dir_y = 0.0
        push_vx = next_obs[8] * 3.0
        push_vy = next_obs[9] * 3.0
        push_align = push_vx * dir_x + push_vy * dir_y
        if push_align < 0.0:
            push_align = 0.0
        if push_align > 1.5:
            push_align = 1.5
        push_reward = 1.5 * push_align
    else:
        push_reward = 0.0

    # ---- 组件 4: dock_inside_alignment ----
    # 关键修复：alignment 只在货箱"真正接近 dock 中心"时才给分
    # 门控从 4.0m 收紧到 1.2m，且用陡峭衰减，避免远距离刷分
    if nxt_dist < 1.2:
        align_gate = 1.0 - nxt_dist / 1.2
        align_gate = align_gate * align_gate
        alignment_reward = 4.0 * align_gate * align_cos
    else:
        alignment_reward = 0.0

    # ---- 组件 5: dock_inside_settling ----
    # 关键修复：settling 只在货箱"真正接近 dock 中心"时才给分
    # 门控从 3.0m 收紧到 1.0m，且用陡峭衰减
    if nxt_dist < 1.0:
        settle_gate = 1.0 - nxt_dist / 1.0
        settle_gate = settle_gate * settle_gate
        speed_factor = 1.0 / (1.0 + 6.0 * nxt_crate_speed)
        settling_reward = 4.0 * settle_gate * speed_factor
    else:
        settling_reward = 0.0

    # ---- 组件 6: joint_condition_proxy（联合完成近似） ----
    # 门控收紧到 0.8m（真正在 dock 内），用几何平均避免塌缩
    if nxt_dist < 0.8:
        pos_factor = 1.0 / (1.0 + 8.0 * nxt_dist)
        align_factor = align_cos
        vel_factor = 1.0 / (1.0 + 25.0 * nxt_crate_speed)
        joint_proxy = (pos_factor * align_factor * vel_factor) ** (1.0 / 3.0)
        joint_reward = 3.0 * joint_proxy
    else:
        joint_reward = 0.0

    # ---- 组件 7: impact_softness ----
    impact_penalty = 0.0
    if contact > 0.5:
        crate_speed_jump = nxt_crate_speed - cur_crate_speed
        if crate_speed_jump > 0.0:
            impact_penalty -= 0.5 * crate_speed_jump
        if nxt_crate_speed > 1.0:
            impact_penalty -= 0.3 * (nxt_crate_speed - 1.0)

    # ---- 组件 8: boundary_avoidance ----
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    boundary_penalty = 0.0
    if abs(cart_x) > 0.85:
        boundary_penalty -= 0.5 * (abs(cart_x) - 0.85)
    if abs(cart_y) > 0.85:
        boundary_penalty -= 0.5 * (abs(cart_y) - 0.85)
    if front_prox > 0.8:
        boundary_penalty -= 0.3 * (front_prox - 0.8)

    cart_cos = next_obs[2]
    cart_sin = next_obs[3]
    rel_bx = next_obs[6] * 3.0
    rel_by = next_obs[7] * 3.0
    crate_wx = cart_cos * rel_bx - cart_sin * rel_by + cart_x * 5.0
    crate_wy = cart_sin * rel_bx + cart_cos * rel_by + cart_y * 4.0
    if abs(crate_wx) > 4.3:
        boundary_penalty -= 0.5 * (abs(crate_wx) - 4.3)
    if abs(crate_wy) > 3.4:
        boundary_penalty -= 0.5 * (abs(crate_wy) - 3.4)

    # ---- 组件 9: action_smoothness ----
    drive = action[0]
    steer = action[1]
    action_penalty = -0.02 * (drive * drive + steer * steer)

    # ---- 总奖励 ----
    components = {}
    components["crate_to_dock_progress"] = float(progress_reward)
    components["dock_proximity"] = float(proximity_reward)
    components["contact_gated_push"] = float(push_reward)
    components["dock_inside_alignment"] = float(alignment_reward)
    components["dock_inside_settling"] = float(settling_reward)
    components["joint_condition_proxy"] = float(joint_reward)
    components["impact_softness"] = float(impact_penalty)
    components["boundary_avoidance"] = float(boundary_penalty)
    components["action_smoothness"] = float(action_penalty)

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return float(total_reward), components