def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 观测字段读取（严格按声明索引） ----
    cart_x = obs[0]
    cart_y = obs[1]
    cart_fwd_speed = obs[4]
    crate_rel_x_body = obs[6]
    crate_rel_y_body = obs[7]
    crate_vx = obs[8]
    crate_vy = obs[9]
    crate_cos = obs[10]
    crate_sin = obs[11]
    dock_dx = obs[12]
    dock_dy = obs[13]
    contact = obs[14]
    sensor_front = obs[15]
    sensor_left = obs[16]
    sensor_right = obs[17]
    time_fraction = obs[18]

    next_dock_dx = next_obs[12]
    next_dock_dy = next_obs[13]
    next_crate_cos = next_obs[10]
    next_crate_sin = next_obs[11]
    next_crate_vx = next_obs[8]
    next_crate_vy = next_obs[9]

    # ---- 货箱到泊位归一化距离 ----
    dist_now = (dock_dx * dock_dx + dock_dy * dock_dy) ** 0.5
    dist_next = (next_dock_dx * next_dock_dx + next_dock_dy * next_dock_dy) ** 0.5

    # ---- 组件 1：货箱向泊位推进（delta 形式，避免悬停陷阱） ----
    progress = dist_now - dist_next
    crate_to_dock_progress = 8.0 * progress

    # ---- 组件 2：货箱位置接近泊位（有界，防悬停） ----
    dock_proximity = 1.0 / (1.0 + 6.0 * dist_now)
    crate_dock_proximity = 1.2 * dock_proximity

    # ---- 组件 3：货箱朝向对齐（cos 误差） ----
    heading_err = 1.0 - crate_cos
    crate_heading_align = 0.8 * (1.0 / (1.0 + 4.0 * heading_err))

    # ---- 组件 4：接近泊位时的货箱速度抑制 ----
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    near_dock = 1.0 / (1.0 + 8.0 * dist_now)
    crate_speed_near_dock = -3.0 * near_dock * crate_speed

    # ---- 组件 5：联合停靠质量代理（位置 + 朝向 + 静止） ----
    f_pos = 1.0 / (1.0 + 10.0 * dist_now)
    f_head = 1.0 / (1.0 + 4.0 * heading_err)
    f_still = 1.0 / (1.0 + 20.0 * crate_speed)
    dock_quality = 2.0 * (f_pos * f_head * f_still) ** (1.0 / 3.0)

    # ---- 组件 6：软接触惩罚（接触时推箱速度过大） ----
    push_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    soft_contact = -1.5 * contact * push_speed

    # ---- 组件 7：越界风险（小车与货箱接近仓库边界） ----
    cart_edge = max(0.0, abs(cart_x) - 0.85) + max(0.0, abs(cart_y) - 0.85)
    dock_edge = max(0.0, abs(dock_dx) - 0.85) + max(0.0, abs(dock_dy) - 0.85)
    out_of_bounds = -4.0 * (cart_edge + dock_edge)

    # ---- 组件 8：障碍接近惩罚 ----
    obstacle_penalty = -0.4 * (sensor_front * sensor_front
                               + sensor_left * sensor_left
                               + sensor_right * sensor_right)

    # ---- 组件 9：动作平滑（轻量） ----
    action_smoothness = -0.05 * (action[0] * action[0] + action[1] * action[1])

    # ---- 组件 10：时间压力（后期轻微惩罚） ----
    time_pressure = -0.3 * time_fraction * time_fraction

    components = {
        "crate_to_dock_progress": crate_to_dock_progress,
        "crate_dock_proximity": crate_dock_proximity,
        "crate_heading_align": crate_heading_align,
        "crate_speed_near_dock": crate_speed_near_dock,
        "dock_quality": dock_quality,
        "soft_contact": soft_contact,
        "out_of_bounds": out_of_bounds,
        "obstacle_penalty": obstacle_penalty,
        "action_smoothness": action_smoothness,
        "time_pressure": time_pressure,
    }

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return float(total_reward), components