def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 货箱到泊位距离（归一化量纲） ----------
    # obs[12]: 货箱中心到泊位中心的有符号 x 偏移 / 仓库半宽
    # obs[13]: 货箱中心到泊位中心的有符号 y 偏移 / 仓库半高
    cur_dx = obs[12]
    cur_dy = obs[13]
    nxt_dx = next_obs[12]
    nxt_dy = next_obs[13]

    cur_dist = (cur_dx * cur_dx + cur_dy * cur_dy) ** 0.5
    nxt_dist = (nxt_dx * nxt_dx + nxt_dy * nxt_dy) ** 0.5

    # ---------- 主职责 1：货箱向泊位推进（改进量，避免悬停陷阱） ----------
    progress = (cur_dist - nxt_dist)
    # 限制单步改进量，防止被大跳变刷分
    if progress > 0.5:
        progress = 0.5
    if progress < -0.5:
        progress = -0.5
    crate_progress = 10.0 * progress

    # ---------- 主职责 2：停靠质量（位置进入 + 朝向对齐 + 静止） ----------
    # 位置因子：越接近泊位中心越接近 1
    pos_factor = 1.0 / (1.0 + 12.0 * nxt_dist)

    # 朝向因子：货箱朝向归一化误差
    crate_cos = next_obs[10]
    crate_sin = next_obs[11]
    norm = (crate_cos * crate_cos + crate_sin * crate_sin) ** 0.5
    if norm < 1e-6:
        norm = 1e-6
    crate_cos_n = crate_cos / norm
    crate_sin_n = crate_sin / norm
    # 目标朝向为 0 弧度 => cos 应接近 1
    align_err = 1.0 - crate_cos_n
    if align_err < 0.0:
        align_err = 0.0
    align_factor = 1.0 / (1.0 + 8.0 * align_err)

    # 静止因子：货箱世界系速度（obs[8], obs[9] 已除以 3.0 m/s）
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    # 速度 0.05 m/s 视为静止；用平滑衰减
    speed_factor = 1.0 / (1.0 + 6.0 * crate_speed)

    # 联合条件（几何平均，缓解塌缩）
    dock_quality = (pos_factor * align_factor * speed_factor) ** (1.0 / 3.0)
    # 仅在货箱足够接近泊位时才强化该信号，避免远处就被对齐项误导
    near_gate = 1.0 / (1.0 + 6.0 * nxt_dist)
    crate_docking = 6.0 * dock_quality * near_gate

    # ---------- 主职责 2b：进入泊位 + 静止的稀疏加成（软） ----------
    in_dock = 1.0
    if abs(nxt_dx) > 0.024:
        in_dock = 0.0
    if abs(nxt_dy) > 0.030:
        in_dock = 0.0
    still = 1.0 / (1.0 + 20.0 * crate_speed)
    aligned = 1.0 / (1.0 + 10.0 * align_err)
    dock_hold_bonus = 3.0 * in_dock * still * aligned

    # ---------- 条件职责：接近泊位时抑制货箱速度（防滑过） ----------
    # 仅在货箱接近泊位时启用
    if nxt_dist < 0.15:
        speed_pen = -1.5 * crate_speed * (0.15 - nxt_dist) / 0.15
    else:
        speed_pen = 0.0

    # ---------- 条件职责：软接触惩罚（间接推断硬碰撞） ----------
    # 接触时货箱速度突增视为硬推/冲击
    contact = next_obs[14]
    if contact > 0.5:
        # 速度突变：当前货箱速度相对上一时刻
        prev_vx = obs[8] * 3.0
        prev_vy = obs[9] * 3.0
        prev_speed = (prev_vx * prev_vx + prev_vy * prev_vy) ** 0.5
        dv = crate_speed - prev_speed
        if dv > 0.0:
            # 推动中货箱加速过快 -> 疑似硬碰撞
            soft_contact = -0.8 * dv
        else:
            soft_contact = 0.0
    else:
        soft_contact = 0.0

    # ---------- 条件职责：越界惩罚（hinge，边界附近才生效） ----------
    # 小车位置 obs[0] (0=中线, +1=远墙), obs[1] (0=中线)
    out_pen = 0.0
    cart_x = obs[0]
    cart_y = obs[1]
    # 以 |x|>0.9 或 |y|>0.9 视为接近边界
    if abs(cart_x) > 0.9:
        out_pen -= 1.0 * (abs(cart_x) - 0.9) / 0.1
    if abs(cart_y) > 0.9:
        out_pen -= 1.0 * (abs(cart_y) - 0.9) / 0.1
    # 货箱到泊位偏移过大（远离中心）也提示越界风险
    if cur_dist > 1.2:
        out_pen -= 0.5 * (cur_dist - 1.2)

    # ---------- 可选：动作平滑（轻量） ----------
    act_drive = action[0]
    act_steer = action[1]
    action_smooth = -0.05 * (act_drive * act_drive + act_steer * act_steer)

    # ---------- 障碍接近惩罚（前/左/右） ----------
    sensor_front = obs[15]
    sensor_left = obs[16]
    sensor_right = obs[17]
    sensor_pen = 0.0
    if sensor_front > 0.8:
        sensor_pen -= 0.3 * (sensor_front - 0.8) / 0.2
    if sensor_left > 0.8:
        sensor_pen -= 0.2 * (sensor_left - 0.8) / 0.2
    if sensor_right > 0.8:
        sensor_pen -= 0.2 * (sensor_right - 0.8) / 0.2

    components = {
        "crate_progress": crate_progress,
        "crate_docking_quality": crate_docking,
        "dock_hold_bonus": dock_hold_bonus,
        "speed_near_dock_penalty": speed_pen,
        "soft_contact_penalty": soft_contact,
        "out_of_bounds_penalty": out_pen,
        "action_smoothness": action_smooth,
        "obstacle_proximity_penalty": sensor_pen,
    }

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return (float(total_reward), components)