def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 读取信号（严格按声明的索引） ----
    crate_vx = obs[8] * 3.0
    crate_vy = obs[9] * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    n_crate_vx = next_obs[8] * 3.0
    n_crate_vy = next_obs[9] * 3.0
    n_crate_speed = (n_crate_vx * n_crate_vx + n_crate_vy * n_crate_vy) ** 0.5

    dx = obs[12]
    dy = obs[13]
    n_dx = next_obs[12]
    n_dy = next_obs[13]

    dist = (dx * dx + dy * dy) ** 0.5
    n_dist = (n_dx * n_dx + n_dy * n_dy) ** 0.5

    # 货箱朝向误差（弧度），cos/sin 归一化后取夹角
    ch = obs[10]
    sh = obs[11]
    norm_h = (ch * ch + sh * sh) ** 0.5
    if norm_h < 1e-6:
        cos_err = 1.0
    else:
        cos_err = ch / norm_h
    if cos_err > 1.0:
        cos_err = 1.0
    if cos_err < -1.0:
        cos_err = -1.0
    angle_err = (1.0 - cos_err) ** 0.5 * 1.41421356  # ~ |angle| for small angles

    # 泊位几何阈值（来自环境卡片）
    inside_x = abs(dx) <= 0.024
    inside_y = abs(dy) <= 0.030
    inside = 1.0 if (inside_x and inside_y) else 0.0

    contact = obs[14]
    sensor_front = obs[15]
    cart_x = obs[0]
    cart_y = obs[1]

    # ---- 主信号 1：货箱到泊位距离的改进（delta） ----
    progress = dist - n_dist  # >0 表示靠近泊位
    crate_to_dock_progress = 60.0 * progress

    # 接近泊位时给进度额外加权，鼓励真正进入
    near_factor = 1.0 / (1.0 + 3.0 * dist)
    crate_to_dock_progress += 25.0 * progress * near_factor

    # ---- 主信号 2：泊位质量（联合条件代理） ----
    # 位置因子：越接近泊位中心越大
    pos_factor = 1.0 / (1.0 + 40.0 * dist)
    # 朝向因子：误差 < 30deg (~0.524 rad) 时接近 1
    ang_factor = 1.0 - min(1.0, angle_err / 0.524)
    # 静止因子：速度 < 0.05 时接近 1
    speed_factor = 1.0 - min(1.0, n_crate_speed / 0.05)
    # 几何因子：完全在泊位内才为 1
    geo_factor = inside

    quality = (pos_factor * ang_factor * speed_factor * geo_factor) ** 0.25
    crate_docking_quality = 40.0 * quality

    # 进入泊位的一次性/持续 bonus（稀疏但连续化）
    dock_entry_bonus = 15.0 * inside * ang_factor

    # ---- 条件信号 3：接近泊位时的速度抑制（门控，只在接近时激活） ----
    if dist < 0.15:
        gate = 1.0 - dist / 0.15
        crate_speed_penalty_near_dock = -8.0 * gate * min(1.0, n_crate_speed / 0.05)
    else:
        crate_speed_penalty_near_dock = 0.0

    # ---- 条件信号 4：软接触惩罚（仅在接触且速度突变大时轻罚） ----
    speed_jump = 0.0
    if contact > 0.5:
        speed_jump = abs(n_crate_speed - crate_speed)
    soft_contact_penalty = -3.0 * min(1.0, speed_jump / 1.5)

    # ---- 条件信号 5：越界惩罚（hinge，仅接近边界时） ----
    out_pen = 0.0
    if abs(cart_x) > 0.85:
        out_pen += -5.0 * (abs(cart_x) - 0.85)
    if abs(cart_y) > 0.85:
        out_pen += -5.0 * (abs(cart_y) - 0.85)
    if abs(dx) > 0.9:
        out_pen += -5.0 * (abs(dx) - 0.9)
    if abs(dy) > 0.9:
        out_pen += -5.0 * (abs(dy) - 0.9)
    out_of_bounds_penalty = out_pen

    # ---- 条件信号 6：前方障碍接近惩罚（hinge） ----
    obstacle_penalty = -2.0 * max(0.0, sensor_front - 0.7)

    # ---- 条件信号 7：动作平滑（轻量，避免压制推动） ----
    action_smoothness = -0.3 * (action[0] * action[0] + action[1] * action[1])

    components = {
        "crate_to_dock_progress": float(crate_to_dock_progress),
        "crate_docking_quality": float(crate_docking_quality),
        "dock_entry_bonus": float(dock_entry_bonus),
        "crate_speed_penalty_near_dock": float(crate_speed_penalty_near_dock),
        "soft_contact_penalty": float(soft_contact_penalty),
        "out_of_bounds_penalty": float(out_of_bounds_penalty),
        "obstacle_penalty": float(obstacle_penalty),
        "action_smoothness": float(action_smoothness),
    }

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return float(total_reward), components