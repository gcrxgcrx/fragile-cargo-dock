def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 从观测中恢复任务相关量 ----
    # 货箱到泊位的归一化偏移
    dx = next_obs[12]
    dy = next_obs[13]
    dx_old = obs[12]
    dy_old = obs[13]

    # 归一化距离（用半宽/半高近似等权）
    dist_new = (dx * dx + dy * dy) ** 0.5
    dist_old = (dx_old * dx_old + dy_old * dy_old) ** 0.5

    # 货箱速度（世界系，归一化）
    cvx = next_obs[8]
    cvy = next_obs[9]
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向误差（弧度）
    heading_err = (next_obs[11] * next_obs[11] + next_obs[10] * next_obs[10]) ** 0.5
    if heading_err < 1e-6:
        heading_err = 1e-6
    cos_err = next_obs[10] / heading_err
    if cos_err > 1.0:
        cos_err = 1.0
    if cos_err < -1.0:
        cos_err = -1.0
    # 朝向对齐度：1 表示完全对齐，0 表示 90 度
    align = cos_err

    # 接触标志
    contact = next_obs[14]

    # 障碍接近度
    front = next_obs[15]
    left = next_obs[16]
    right = next_obs[17]

    # 小车位置（判断越界风险）
    cart_x = next_obs[0]
    cart_y = next_obs[1]

    # ---- 组件 1：货箱向泊位推进（主信号，delta 形式避免悬停）----
    progress = dist_old - dist_new
    # 放大尺度，使其成为主信号
    crate_to_dock_progress = 6.0 * progress

    # ---- 组件 2：货箱在泊位附近的停靠质量（门控式，仅在接近泊位时激活）----
    # 接近度门：dist 越小越接近 1
    near_gate = 1.0 / (1.0 + 8.0 * dist_new)
    # 速度抑制：接近泊位时速度应下降
    speed_term = -1.5 * crate_speed * near_gate
    # 朝向对齐：接近泊位时朝向应对齐
    align_term = 1.0 * align * near_gate * align
    # 完全进入泊位的连续 proxy（|dx|<=0.024, |dy|<=0.030）
    in_x = 1.0
    if abs(dx) > 0.024:
        in_x = 0.024 / abs(dx)
    in_y = 1.0
    if abs(dy) > 0.030:
        in_y = 0.030 / abs(dy)
    inside_proxy = in_x * in_y
    # 静止且进入且对齐的联合信号
    still = 1.0 / (1.0 + 20.0 * crate_speed)
    dock_quality = 2.0 * inside_proxy * still * (0.5 + 0.5 * align)

    crate_docking_quality = speed_term + align_term + dock_quality

    # ---- 组件 3：轻柔操作（抑制接触时的高速冲撞）----
    # 仅在接触且货箱速度偏高时惩罚，避免误伤正常推动
    hard_contact = contact * crate_speed
    soft_contact_penalty = -0.8 * hard_contact

    # ---- 组件 4：越界风险 hinge 惩罚 ----
    # 小车位置越界（|x|>1 或 |y|>1 为越界，留 15% 余量）
    cart_margin = 0.85
    cart_over = 0.0
    ax = abs(cart_x)
    ay = abs(cart_y)
    if ax > cart_margin:
        cart_over = cart_over + (ax - cart_margin)
    if ay > cart_margin:
        cart_over = cart_over + (ay - cart_margin)
    # 货箱越界风险：货箱到泊位偏移过大意味着远离中心，用泊位偏移近似
    crate_over = 0.0
    if dist_new > 0.9:
        crate_over = dist_new - 0.9
    out_of_bounds_penalty = -3.0 * cart_over - 3.0 * crate_over

    # ---- 组件 5：障碍接近度 hinge 惩罚（仅在很近时生效）----
    obstacle_penalty = 0.0
    if front > 0.8:
        obstacle_penalty = obstacle_penalty - 0.5 * (front - 0.8)
    if left > 0.85:
        obstacle_penalty = obstacle_penalty - 0.3 * (left - 0.85)
    if right > 0.85:
        obstacle_penalty = obstacle_penalty - 0.3 * (right - 0.85)

    # ---- 组件 6：动作平滑（轻量，避免抖动）----
    action_smoothness = -0.02 * (action[0] * action[0] + action[1] * action[1])

    components = {
        "crate_to_dock_progress": crate_to_dock_progress,
        "crate_docking_quality": crate_docking_quality,
        "soft_contact_penalty": soft_contact_penalty,
        "out_of_bounds_penalty": out_of_bounds_penalty,
        "obstacle_penalty": obstacle_penalty,
        "action_smoothness": action_smoothness,
    }

    total_reward = 0.0
    for key in components:
        total_reward = total_reward + components[key]

    return (float(total_reward), components)