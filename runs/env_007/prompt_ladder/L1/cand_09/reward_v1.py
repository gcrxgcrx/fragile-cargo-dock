def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 货箱到泊位偏移（有符号，归一化）----
    dx = next_obs[12]
    dy = next_obs[13]
    pdx = obs[12]
    pdy = obs[13]

    dist = (dx * dx + dy * dy) ** 0.5
    prev_dist = (pdx * pdx + pdy * pdy) ** 0.5

    # ---- 主信号 1：货箱向泊位推进（delta 距离，避免悬停陷阱）----
    progress = prev_dist - dist
    crate_to_dock_progress = 6.0 * progress

    # ---- 主信号 2：货箱停靠质量（位置 + 朝向 + 静止 的联合代理）----
    # 位置因子：距离越小越接近 1
    pos_factor = 1.0 / (1.0 + 8.0 * dist)

    # 朝向因子：货箱朝向误差 -> cos(误差) 近似（货箱朝向即泊位目标朝向）
    crate_hx = next_obs[10]
    crate_hy = next_obs[11]
    align = (crate_hx * crate_hx + crate_hy * crate_hy) ** 0.5
    if align < 1e-6:
        align_cos = 0.0
    else:
        align_cos = crate_hx / align
    orient_factor = (align_cos + 1.0) * 0.5  # 映射到 [0,1]

    # 静止因子：货箱速度越小越接近 1（仅在接近泊位时才有意义，用 pos_factor 门控）
    cvx = next_obs[8] * 3.0
    cvy = next_obs[9] * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    rest_factor = 1.0 / (1.0 + 20.0 * crate_speed)

    # 联合代理：几何平均，避免乘积塌缩
    joint = (pos_factor * orient_factor * rest_factor) ** (1.0 / 3.0)
    # 仅在货箱接近泊位时激活（门控），避免全局持续收分
    near_gate = 1.0 / (1.0 + 12.0 * dist)
    crate_docking_quality = 3.0 * joint * near_gate

    # ---- 条件信号 3：接近泊位时的货箱速度抑制（hinge，仅在近处生效）----
    speed_excess = crate_speed - 0.4
    if speed_excess < 0.0:
        speed_excess = 0.0
    near_prox = 1.0 / (1.0 + 10.0 * dist)
    crate_speed_penalty_near_dock = -2.0 * near_prox * speed_excess

    # ---- 条件信号 4：软接触惩罚（接触时货箱速度过大 -> 可能硬碰撞）----
    contact = next_obs[14]
    if contact > 0.5:
        contact_speed_excess = crate_speed - 0.8
        if contact_speed_excess < 0.0:
            contact_speed_excess = 0.0
        soft_contact_penalty = -1.5 * contact_speed_excess
    else:
        soft_contact_penalty = 0.0

    # ---- 条件信号 5：越界 hinge 惩罚（小车与货箱接近边界）----
    cart_x = obs[0]
    cart_y = obs[1]
    cart_margin = 0.90 - abs(cart_x)
    if cart_margin > 0.10:
        cart_margin = 0.10
    if cart_margin < 0.0:
        cart_margin = 0.0
    cart_pen = (0.10 - cart_margin)
    cart_y_margin = 0.90 - abs(cart_y)
    if cart_y_margin > 0.10:
        cart_y_margin = 0.10
    if cart_y_margin < 0.0:
        cart_y_margin = 0.0
    cart_pen_y = (0.10 - cart_y_margin)

    # 货箱世界坐标近似恢复（用于边界检查）
    ch = obs[2]
    sh = obs[3]
    relx = obs[6] * 3.0
    rely = obs[7] * 3.0
    crate_wx = cart_x * 5.0 + (ch * relx - sh * rely)
    crate_wy = cart_y * 4.0 + (sh * relx + ch * rely)
    crate_margin_x = 0.90 - abs(crate_wx / 5.0)
    if crate_margin_x > 0.10:
        crate_margin_x = 0.10
    if crate_margin_x < 0.0:
        crate_margin_x = 0.0
    crate_pen_x = (0.10 - crate_margin_x)
    crate_margin_y = 0.90 - abs(crate_wy / 4.0)
    if crate_margin_y > 0.10:
        crate_margin_y = 0.10
    if crate_margin_y < 0.0:
        crate_margin_y = 0.0
    crate_pen_y = (0.10 - crate_margin_y)

    out_of_bounds_penalty = -3.0 * (cart_pen + cart_pen_y + crate_pen_x + crate_pen_y)

    # ---- 条件信号 6：动作平滑（轻量，不压制推动）----
    d = action[0]
    s = action[1]
    action_smoothness = -0.05 * (d * d + s * s)

    components = {
        "crate_to_dock_progress": crate_to_dock_progress,
        "crate_docking_quality": crate_docking_quality,
        "crate_speed_penalty_near_dock": crate_speed_penalty_near_dock,
        "soft_contact_penalty": soft_contact_penalty,
        "out_of_bounds_penalty": out_of_bounds_penalty,
        "action_smoothness": action_smoothness,
    }

    total_reward = (
        crate_to_dock_progress
        + crate_docking_quality
        + crate_speed_penalty_near_dock
        + soft_contact_penalty
        + out_of_bounds_penalty
        + action_smoothness
    )

    return (float(total_reward), components)