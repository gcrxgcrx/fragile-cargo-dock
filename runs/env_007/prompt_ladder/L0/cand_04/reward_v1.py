def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 常量 ----------
    # 泊位几何阈值（来自环境卡片）
    DOCK_X_TOL = 0.024   # |obs[12]| <= 0.024 视为 x 方向完全进入
    DOCK_Y_TOL = 0.030   # |obs[13]| <= 0.030 视为 y 方向完全进入
    # 位置尺度（归一化偏移 -> 米）
    HALF_W = 5.0
    HALF_H = 4.0
    # 朝向对齐阈值（30 度）
    ANG_TOL = 0.5236
    # 货箱速度阈值（m/s），obs 中已除以 3.0
    SPEED_TOL = 0.05 / 3.0

    # ---------- 位置误差（米制，用于主进度信号） ----------
    dx_old = obs[12] * HALF_W
    dy_old = obs[13] * HALF_H
    dx_new = next_obs[12] * HALF_W
    dy_new = next_obs[13] * HALF_H

    dist_old = (dx_old * dx_old + dy_old * dy_old) ** 0.5
    dist_new = (dx_new * dx_new + dy_new * dy_new) ** 0.5

    # ---------- 货箱速度（m/s） ----------
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    crate_speed_norm = crate_speed / 3.0

    # ---------- 货箱朝向误差 ----------
    cos_h = next_obs[10]
    sin_h = next_obs[11]
    norm_h = (cos_h * cos_h + sin_h * sin_h) ** 0.5
    if norm_h < 1e-6:
        norm_h = 1e-6
    cos_h = cos_h / norm_h
    sin_h = sin_h / norm_h
    # 货箱朝向角（相对世界 x 轴）
    angle_err = (sin_h * sin_h) ** 0.5  # 用 |sin| 近似偏离对齐的程度
    # 更稳健：使用 1 - cos_h 作为对齐度量（cos_h=1 表示完全对齐）
    align_metric = cos_h  # [-1, 1]，越接近 1 越对齐

    # ---------- 是否接近泊位 ----------
    # 用归一化偏移量判断接近程度
    near_dock = 1.0 / (1.0 + 3.0 * (dx_new * dx_new + dy_new * dy_new) ** 0.5 / 1.0)
    # 更直接的接近因子：距离越近越接近 1
    dist_norm = (dx_new * dx_new + dy_new * dy_new) ** 0.5
    near_factor = 1.0 / (1.0 + 2.0 * dist_norm)

    # ---------- 组件 1：货箱向泊位推进（主信号，delta 形式） ----------
    progress = dist_old - dist_new  # 减少为正
    crate_to_dock_progress = 3.0 * progress

    # ---------- 组件 2：泊位内位置质量（完全进入的连续 proxy） ----------
    # x 方向：误差越小越接近 1
    x_q = 1.0 / (1.0 + 20.0 * (dx_new * dx_new) ** 0.5)
    y_q = 1.0 / (1.0 + 20.0 * (dy_new * dy_new) ** 0.5)
    # 只在货箱接近泊位时才给予位置质量奖励（避免远处刷分）
    pos_quality = near_factor * (x_q + y_q) * 0.5
    crate_position_quality = 1.5 * pos_quality

    # ---------- 组件 3：朝向对齐 shaping ----------
    # align_metric 接近 1 表示对齐；用 (align_metric + 1) / 2 映射到 [0,1]
    align_score = (align_metric + 1.0) * 0.5  # [0,1]
    # 只在货箱接近泊位时强调对齐
    orient_quality = near_factor * align_score
    crate_orientation_quality = 1.0 * orient_quality

    # ---------- 组件 4：接近泊位时抑制货箱速度 ----------
    # 速度惩罚，只在接近泊位时启用
    speed_excess = crate_speed_norm - SPEED_TOL
    if speed_excess < 0.0:
        speed_excess = 0.0
    crate_speed_penalty_near_dock = -2.0 * near_factor * (speed_excess * speed_excess)

    # ---------- 组件 5：联合停靠 proxy（进入 + 对齐 + 静止） ----------
    # 用连续 bounded factor 构造几何平均，避免塌缩
    in_dock_x = 1.0 / (1.0 + 40.0 * (dx_new * dx_new) ** 0.5)
    in_dock_y = 1.0 / (1.0 + 40.0 * (dy_new * dy_new) ** 0.5)
    aligned = 1.0 / (1.0 + 10.0 * (1.0 - align_metric))
    slow = 1.0 / (1.0 + 20.0 * crate_speed_norm)
    # 几何平均
    joint_dock_proxy = (in_dock_x * in_dock_y * aligned * slow) ** 0.25
    docking_joint_bonus = 2.0 * joint_dock_proxy

    # ---------- 组件 6：软接触惩罚（间接推断） ----------
    # 接触时货箱速度突变可能意味着硬碰撞；用接触 + 速度做轻量惩罚
    contact = next_obs[14]
    # 接触时货箱速度过快 -> 可能硬碰撞
    contact_speed_excess = crate_speed_norm - 0.3
    if contact_speed_excess < 0.0:
        contact_speed_excess = 0.0
    soft_contact_penalty = -1.0 * contact * (contact_speed_excess * contact_speed_excess)

    # ---------- 组件 7：越界风险惩罚（hinge） ----------
    # 小车位置归一化到 [-1, 1] 附近，接近 ±1 时惩罚
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    # 货箱世界坐标恢复
    cos_c = next_obs[2]
    sin_c = next_obs[3]
    norm_c = (cos_c * cos_c + sin_c * sin_c) ** 0.5
    if norm_c < 1e-6:
        norm_c = 1e-6
    cos_c = cos_c / norm_c
    sin_c = sin_c / norm_c
    rel_x = next_obs[6] * 3.0
    rel_y = next_obs[7] * 3.0
    crate_wx = cart_x * HALF_W + rel_x * cos_c - rel_y * sin_c
    crate_wy = cart_y * HALF_H + rel_x * sin_c + rel_y * cos_c
    crate_nx = crate_wx / HALF_W
    crate_ny = crate_wy / HALF_H

    # hinge：超出 0.85 边界才开始惩罚
    bound_lim = 0.85
    cart_x_excess = abs(cart_x) - bound_lim
    if cart_x_excess < 0.0:
        cart_x_excess = 0.0
    cart_y_excess = abs(cart_y) - bound_lim
    if cart_y_excess < 0.0:
        cart_y_excess = 0.0
    crate_x_excess = abs(crate_nx) - bound_lim
    if crate_x_excess < 0.0:
        crate_x_excess = 0.0
    crate_y_excess = abs(crate_ny) - bound_lim
    if crate_y_excess < 0.0:
        crate_y_excess = 0.0
    out_of_bounds_penalty = -3.0 * (cart_x_excess + cart_y_excess + crate_x_excess + crate_y_excess)

    # ---------- 组件 8：动作平滑（轻量） ----------
    action_smoothness = -0.05 * (action[0] * action[0] + action[1] * action[1])

    # ---------- 汇总 ----------
    components = {
        "crate_to_dock_progress": float(crate_to_dock_progress),
        "crate_position_quality": float(crate_position_quality),
        "crate_orientation_quality": float(crate_orientation_quality),
        "crate_speed_penalty_near_dock": float(crate_speed_penalty_near_dock),
        "docking_joint_bonus": float(docking_joint_bonus),
        "soft_contact_penalty": float(soft_contact_penalty),
        "out_of_bounds_penalty": float(out_of_bounds_penalty),
        "action_smoothness": float(action_smoothness),
    }

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return (float(total_reward), components)