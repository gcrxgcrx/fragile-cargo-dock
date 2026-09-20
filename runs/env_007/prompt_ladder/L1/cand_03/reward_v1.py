def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 主信号：货箱到泊位中心的距离缩减（delta / improvement） ----
    dx_old = obs[12]
    dy_old = obs[13]
    dx_new = next_obs[12]
    dy_new = next_obs[13]

    dist_old = (dx_old * dx_old + dy_old * dy_old) ** 0.5
    dist_new = (dx_new * dx_new + dy_new * dy_new) ** 0.5

    # 用有符号加权距离（x 方向权重更大，泊位 x 容差更紧）
    wdist_old = (2.0 * dx_old * dx_old + dy_old * dy_old) ** 0.5
    wdist_new = (2.0 * dx_new * dx_new + dy_new * dy_new) ** 0.5

    progress = wdist_old - wdist_new  # 正=靠近泊位

    # ---- 泊位接近度门（连续，0~1），用于激活"停靠质量"类信号 ----
    # 当 |dx|<=0.024 且 |dy|<=0.030 视为完全进入；用平滑接近度衡量
    near_x = max(0.0, 1.0 - abs(dx_new) / 0.15)
    near_y = max(0.0, 1.0 - abs(dy_new) / 0.18)
    near_gate = near_x * near_y  # 0~1，越接近泊位越大

    # ---- 朝向对齐：货箱朝向与目标朝向（假设泊位朝向为 0，即 cos=1,sin=0）----
    crate_cos = next_obs[10]
    crate_sin = next_obs[11]
    # 朝向误差度量：1 - cos(theta) 在 [0,2]，越小越对齐
    align_factor = max(0.0, (crate_cos + 1.0) * 0.5)  # 1=完全对齐, 0=反向

    # ---- 货箱速度（世界系）----
    cvx = next_obs[8] * 3.0
    cvy = next_obs[9] * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---- 组件 1：货箱向泊位推进（主信号）----
    # 缩放：progress 量级 ~0.01，放大到 ~1
    crate_dock_progress = 60.0 * progress

    # ---- 组件 2：泊位内停靠质量（门控联合条件）----
    # 仅在接近泊位时激活；对齐 + 低速 联合
    speed_factor = max(0.0, 1.0 - crate_speed / 0.5)  # 0.5 m/s 以上为 0
    dock_quality = near_gate * (0.5 * align_factor + 0.5 * speed_factor)
    crate_docking_quality = 3.0 * dock_quality

    # ---- 组件 3：接近泊位时的速度抑制（仅在 near_gate 高时生效）----
    # 高速惩罚只乘在 near_gate 上，避免阻碍远距离运输
    crate_speed_penalty = -1.5 * near_gate * min(1.0, crate_speed / 0.5)

    # ---- 组件 4：软接触惩罚（接触时货箱速度突变过大视为硬推）----
    contact = next_obs[14]
    # 接触时若货箱速度过大，视为硬碰撞风险
    contact_speed = crate_speed
    soft_contact_penalty = -0.8 * contact * min(1.0, max(0.0, (contact_speed - 0.3) / 0.7))

    # ---- 组件 5：越界惩罚（小车与货箱接近仓库边界）----
    cart_x = obs[0]
    cart_y = obs[1]
    cart_oob = max(0.0, abs(cart_x) - 0.92) + max(0.0, abs(cart_y) - 0.92)
    # 货箱世界坐标恢复：crate_rel_body 旋转到世界系
    cos_h = obs[2]
    sin_h = obs[3]
    rel_x = obs[6] * 3.0
    rel_y = obs[7] * 3.0
    crate_wx = rel_x * cos_h - rel_y * sin_h + cart_x * 5.0
    crate_wy = rel_x * sin_h + rel_y * cos_h + cart_y * 4.0
    crate_oob = max(0.0, abs(crate_wx) / 5.0 - 0.92) + max(0.0, abs(crate_wy) / 4.0 - 0.92)
    out_of_bounds_penalty = -4.0 * (cart_oob + crate_oob)

    # ---- 组件 6：动作平滑（轻量，防止抖动）----
    action_smoothness = -0.05 * (action[0] * action[0] + action[1] * action[1])

    components = {
        "crate_dock_progress": crate_dock_progress,
        "crate_docking_quality": crate_docking_quality,
        "crate_speed_penalty": crate_speed_penalty,
        "soft_contact_penalty": soft_contact_penalty,
        "out_of_bounds_penalty": out_of_bounds_penalty,
        "action_smoothness": action_smoothness,
    }

    total_reward = (
        crate_dock_progress
        + crate_docking_quality
        + crate_speed_penalty
        + soft_contact_penalty
        + out_of_bounds_penalty
        + action_smoothness
    )

    return (float(total_reward), components)