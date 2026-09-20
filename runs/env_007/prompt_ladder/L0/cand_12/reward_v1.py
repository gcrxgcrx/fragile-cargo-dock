def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 读取当前与下一步的货箱-泊位偏移 ----
    dx_cur = obs[12]
    dy_cur = obs[13]
    dx_next = next_obs[12]
    dy_next = next_obs[13]

    dist_cur = (dx_cur * dx_cur + dy_cur * dy_cur) ** 0.5
    dist_next = (dx_next * dx_next + dy_next * dy_next) ** 0.5

    # ---- 1) 货箱向泊位的进展信号（主信号，用 delta 避免悬停陷阱）----
    progress = dist_cur - dist_next
    crate_to_dock_progress = 12.0 * progress

    # ---- 2) 泊位内停靠质量（连续联合条件代理）----
    # 位置进入度：货箱完全进入泊位 |dx|<=0.024, |dy|<=0.030
    inside_x = max(0.0, 1.0 - abs(dx_next) / 0.024)
    inside_y = max(0.0, 1.0 - abs(dy_next) / 0.030)
    pos_factor = (inside_x * inside_y) ** 0.5

    # 朝向对齐度：朝向误差 < 30deg (0.5236 rad)
    crate_cos = next_obs[10]
    crate_sin = next_obs[11]
    heading_err = abs(crate_cos - 1.0) + abs(crate_sin)
    align_factor = max(0.0, 1.0 - heading_err / 0.5236)

    # 静止度：速度 < 0.05 m/s
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    still_factor = max(0.0, 1.0 - crate_speed / 0.05)

    dock_quality = 6.0 * (pos_factor * align_factor * still_factor) ** (1.0 / 3.0)
    crate_docking_quality = dock_quality

    # ---- 3) 接近泊位时的速度抑制（只在接近泊位时启用）----
    near = max(0.0, 1.0 - dist_next / 0.30)
    crate_speed_penalty_near_dock = -2.0 * near * (crate_speed / 3.0) ** 2

    # ---- 4) 软接触：接触时抑制极端相对速度突变（间接推断，轻量）----
    contact = next_obs[14]
    cart_speed = abs(next_obs[4])
    crate_speed_norm = crate_speed / 3.0
    rel_harsh = max(0.0, crate_speed_norm - 0.5) + max(0.0, cart_speed - 0.5)
    soft_contact_penalty = -1.5 * contact * rel_harsh

    # ---- 5) 越界防护（hinge，仅在接近边界时生效）----
    cart_x = obs[0]
    cart_y = obs[1]
    over_cart = max(0.0, abs(cart_x) - 0.90) + max(0.0, abs(cart_y) - 0.90)
    over_crate = max(0.0, abs(dx_cur) - 1.20) + max(0.0, abs(dy_cur) - 1.20)
    out_of_bounds_penalty = -8.0 * (over_cart + over_crate)

    # ---- 6) 动作平滑（轻量，避免抖动）----
    action_smoothness = -0.05 * (action[0] ** 2 + action[1] ** 2)

    components = {
        "crate_to_dock_progress": crate_to_dock_progress,
        "crate_docking_quality": crate_docking_quality,
        "crate_speed_penalty_near_dock": crate_speed_penalty_near_dock,
        "soft_contact_penalty": soft_contact_penalty,
        "out_of_bounds_penalty": out_of_bounds_penalty,
        "action_smoothness": action_smoothness,
    }

    total_reward = 0.0
    for key in components:
        total_reward = total_reward + components[key]

    return (float(total_reward), components)