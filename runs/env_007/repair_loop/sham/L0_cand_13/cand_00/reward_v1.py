def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ================= 信号提取 =================
    # 货箱到泊位的归一化偏移
    cur_dx = obs[12]
    cur_dy = obs[13]
    nxt_dx = next_obs[12]
    nxt_dy = next_obs[13]

    cur_dist = (cur_dx * cur_dx + cur_dy * cur_dy) ** 0.5
    nxt_dist = (nxt_dx * nxt_dx + nxt_dy * nxt_dy) ** 0.5

    # 货箱世界系速度（obs[8],obs[9] 已除以 3.0 m/s）
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    # 货箱朝向对齐误差（目标朝向 0 rad => cos 接近 1）
    crate_cos = next_obs[10]
    crate_sin = next_obs[11]
    norm = (crate_cos * crate_cos + crate_sin * crate_sin) ** 0.5
    if norm < 1e-6:
        norm = 1e-6
    crate_cos_n = crate_cos / norm
    align_err = 1.0 - crate_cos_n
    if align_err < 0.0:
        align_err = 0.0

    # ================= 主职责 1：货箱向泊位推进（delta，避免悬停） =================
    progress = cur_dist - nxt_dist
    if progress > 0.5:
        progress = 0.5
    if progress < -0.5:
        progress = -0.5
    crate_progress = 12.0 * progress

    # ================= 主职责 2：靠近泊位的连续塑形（只奖励接近，不惩罚速度） =================
    # 距离越近越大，但有界；这是纯位置信号，不因速度下降
    proximity = 1.0 / (1.0 + 6.0 * nxt_dist)
    crate_proximity = 1.0 * proximity

    # ================= 主职责 3：停靠质量（进入 + 对齐 + 静止的联合门控） =================
    # 二值进入门（几何阈值来自环境卡片）
    in_dock = 1.0
    if abs(nxt_dx) > 0.024:
        in_dock = 0.0
    if abs(nxt_dy) > 0.030:
        in_dock = 0.0

    # 连续因子
    align_factor = 1.0 / (1.0 + 6.0 * align_err)          # 对齐度
    still_factor = 1.0 / (1.0 + 8.0 * crate_speed)        # 静止度

    # 只有在货箱进入泊位后才给停靠质量分；否则为 0
    crate_docking_quality = 8.0 * in_dock * align_factor * still_factor

    # ================= 条件职责：接近泊位时的速度抑制（hinge，仅近距生效） =================
    # 仅在 nxt_dist < 0.12 且货箱速度超过 0.15 m/s 时惩罚过快，避免滑过
    speed_near_dock_penalty = 0.0
    if nxt_dist < 0.12:
        if crate_speed > 0.15:
            speed_near_dock_penalty = -2.0 * (crate_speed - 0.15) * (0.12 - nxt_dist) / 0.12

    # ================= 条件职责：软接触惩罚（仅在接触且货箱加速过快时） =================
    contact = next_obs[14]
    soft_contact_penalty = 0.0
    if contact > 0.5:
        prev_vx = obs[8] * 3.0
        prev_vy = obs[9] * 3.0
        prev_speed = (prev_vx * prev_vx + prev_vy * prev_vy) ** 0.5
        dv = crate_speed - prev_speed
        if dv > 0.3:
            soft_contact_penalty = -1.0 * (dv - 0.3)

    # ================= 条件职责：越界惩罚（hinge，边界附近才生效） =================
    out_of_bounds_penalty = 0.0
    cart_x = obs[0]
    cart_y = obs[1]
    if abs(cart_x) > 0.92:
        out_of_bounds_penalty -= 2.0 * (abs(cart_x) - 0.92) / 0.08
    if abs(cart_y) > 0.92:
        out_of_bounds_penalty -= 2.0 * (abs(cart_y) - 0.92) / 0.08
    # 货箱远离泊位过远提示越界风险（hinge）
    if cur_dist > 1.2:
        out_of_bounds_penalty -= 1.0 * (cur_dist - 1.2)

    # ================= 条件职责：障碍接近惩罚（hinge） =================
    obstacle_proximity_penalty = 0.0
    sensor_front = obs[15]
    sensor_left = obs[16]
    sensor_right = obs[17]
    if sensor_front > 0.85:
        obstacle_proximity_penalty -= 0.5 * (sensor_front - 0.85) / 0.15
    if sensor_left > 0.9:
        obstacle_proximity_penalty -= 0.3 * (sensor_left - 0.9) / 0.1
    if sensor_right > 0.9:
        obstacle_proximity_penalty -= 0.3 * (sensor_right - 0.9) / 0.1

    # ================= 可选：动作平滑（轻量） =================
    act_drive = action[0]
    act_steer = action[1]
    action_smoothness = -0.03 * (act_drive * act_drive + act_steer * act_steer)

    # ================= 汇总 =================
    components = {
        "crate_progress": crate_progress,
        "crate_proximity": crate_proximity,
        "crate_docking_quality": crate_docking_quality,
        "speed_near_dock_penalty": speed_near_dock_penalty,
        "soft_contact_penalty": soft_contact_penalty,
        "out_of_bounds_penalty": out_of_bounds_penalty,
        "obstacle_proximity_penalty": obstacle_proximity_penalty,
        "action_smoothness": action_smoothness,
    }

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return (float(total_reward), components)