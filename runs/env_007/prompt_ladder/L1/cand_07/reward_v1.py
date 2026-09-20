def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 从观测中提取信号（仅使用已声明维度） ----
    # 货箱到泊位的有符号偏移（归一化）
    dx_now = obs[12]
    dy_now = obs[13]
    dx_next = next_obs[12]
    dy_next = next_obs[13]

    # 货箱到泊位的归一化距离（欧氏）
    dist_now = (dx_now * dx_now + dy_now * dy_now) ** 0.5
    dist_next = (dx_next * dx_next + dy_next * dy_next) ** 0.5

    # 货箱速度（世界系，归一化）
    cvx = next_obs[8]
    cvy = next_obs[9]
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向误差（弧度，归一化到 [0, pi]）
    crate_theta = (next_obs[11] * next_obs[11] + next_obs[10] * next_obs[10]) ** 0.5
    if crate_theta < 1e-6:
        angle_err = 0.0
    else:
        # atan2 的近似：用 asin(clip) 不可靠，改用 cos 值近似误差
        # 朝向对齐时 cos≈1；误差越大 cos 越小
        cos_align = next_obs[10] / crate_theta
        if cos_align > 1.0:
            cos_align = 1.0
        if cos_align < -1.0:
            cos_align = -1.0
        # 用 (1 - cos) 作为朝向误差的单调代理，[0, 2]
        angle_err = 1.0 - cos_align

    # 接触标志
    contact = next_obs[14]

    # 障碍接近度
    sensor_front = next_obs[15]
    sensor_left = next_obs[16]
    sensor_right = next_obs[17]

    # 小车位置（归一化，仓库半宽/半高约 1.0 边界）
    cart_x = next_obs[0]
    cart_y = next_obs[1]

    # ---- 组件 1：货箱向泊位推进（主信号，improvement_delta） ----
    # 用距离减少量作为主进展信号，避免悬停陷阱
    progress = dist_now - dist_next
    # 放大到合理量级
    crate_progress = 3.0 * progress

    # ---- 组件 2：接近泊位后的停靠质量（门控：仅当接近泊位时激活） ----
    # 接近门：距离 < 0.15 时开始激活
    if dist_next < 0.30:
        near_gate = (0.30 - dist_next) / 0.30
    else:
        near_gate = 0.0
    if near_gate < 0.0:
        near_gate = 0.0
    if near_gate > 1.0:
        near_gate = 1.0

    # 2a 位置收敛：距离越小越好（仅在接近时）
    pos_quality = near_gate * (1.0 - dist_next)

    # 2b 朝向对齐：cos 越接近 1 越好（仅在接近时）
    align_quality = near_gate * (1.0 - angle_err)

    # 2c 静止：速度越小越好（仅在接近时），用 bounded 形式
    speed_quality = near_gate * (1.0 / (1.0 + 20.0 * crate_speed))

    docking_quality = 0.4 * pos_quality + 0.3 * align_quality + 0.3 * speed_quality

    # ---- 组件 3：接近泊位时的速度抑制（门控，避免滑过泊位） ----
    # 仅在接近泊位且货箱速度较大时惩罚
    if dist_next < 0.25:
        overspeed = crate_speed - 0.05
        if overspeed < 0.0:
            overspeed = 0.0
        speed_penalty = -0.5 * overspeed * overspeed
    else:
        speed_penalty = 0.0

    # ---- 组件 4：软接触惩罚（仅在接触且相对速度较大时） ----
    # 用货箱速度作为间接代理：接触时货箱速度大说明推动较猛
    if contact > 0.5:
        hard_proxy = crate_speed - 0.15
        if hard_proxy < 0.0:
            hard_proxy = 0.0
        soft_contact_penalty = -0.3 * hard_proxy * hard_proxy
    else:
        soft_contact_penalty = 0.0

    # ---- 组件 5：越界惩罚（hinge，仅在接近边界时） ----
    bound_penalty = 0.0
    # 小车边界（归一化坐标，边界约 ±1.0）
    cart_margin = 0.85
    if cart_x > cart_margin:
        bound_penalty -= 0.5 * (cart_x - cart_margin) * (cart_x - cart_margin)
    if cart_x < -cart_margin:
        bound_penalty -= 0.5 * (cart_x + cart_margin) * (cart_x + cart_margin)
    if cart_y > cart_margin:
        bound_penalty -= 0.5 * (cart_y - cart_margin) * (cart_y - cart_margin)
    if cart_y < -cart_margin:
        bound_penalty -= 0.5 * (cart_y + cart_margin) * (cart_y + cart_margin)

    # ---- 组件 6：障碍接近惩罚（hinge，仅在前方接近障碍时） ----
    obstacle_penalty = 0.0
    if sensor_front > 0.7:
        obstacle_penalty -= 0.3 * (sensor_front - 0.7) * (sensor_front - 0.7)
    if sensor_left > 0.8:
        obstacle_penalty -= 0.2 * (sensor_left - 0.8) * (sensor_left - 0.8)
    if sensor_right > 0.8:
        obstacle_penalty -= 0.2 * (sensor_right - 0.8) * (sensor_right - 0.8)

    # ---- 组件 7：动作平滑（轻量，避免抖动） ----
    action_smoothness = -0.02 * (action[0] * action[0] + action[1] * action[1])

    # ---- 汇总 ----
    components = {
        "crate_progress": crate_progress,
        "docking_quality": docking_quality,
        "speed_penalty": speed_penalty,
        "soft_contact_penalty": soft_contact_penalty,
        "bound_penalty": bound_penalty,
        "obstacle_penalty": obstacle_penalty,
        "action_smoothness": action_smoothness,
    }

    total_reward = (
        crate_progress
        + docking_quality
        + speed_penalty
        + soft_contact_penalty
        + bound_penalty
        + obstacle_penalty
        + action_smoothness
    )

    return (float(total_reward), components)