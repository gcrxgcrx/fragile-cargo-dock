def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 几何常量（来自环境事实） ----------
    # 泊位容差：|obs[12]| <= 0.024, |obs[13]| <= 0.030
    dock_tol_x = 0.024
    dock_tol_y = 0.030
    # 朝向误差 < 30 度
    angle_tol = 0.5235987755982988  # 30 deg in rad
    # 速度阈值 < 0.05 m/s（归一化：obs[8],obs[9] 已 /3.0）
    speed_tol = 0.05 / 3.0

    # ---------- 当前帧量 ----------
    dx_cur = obs[12]
    dy_cur = obs[13]
    dx_nxt = next_obs[12]
    dy_nxt = next_obs[13]

    # 归一化到各自容差尺度的距离度量
    d_cur = ((dx_cur / dock_tol_x) ** 2 + (dy_cur / dock_tol_y) ** 2) ** 0.5
    d_nxt = ((dx_nxt / dock_tol_x) ** 2 + (dy_nxt / dock_tol_y) ** 2) ** 0.5

    # 货箱速度（归一化单位）
    vx_nxt = next_obs[8]
    vy_nxt = next_obs[9]
    crate_speed = ((vx_nxt * 3.0) ** 2 + (vy_nxt * 3.0) ** 2) ** 0.5

    # 货箱朝向误差
    crate_ang = (next_obs[11] ** 2 + next_obs[10] ** 2) ** 0.5
    if crate_ang < 1e-6:
        ang_err = 0.0
    else:
        ang_err = (next_obs[11] / crate_ang)
        ang_err = abs(ang_err)
        # |sin(theta)| 作为朝向误差代理，0 对齐，1 垂直
        ang_err = min(1.0, ang_err)

    # 接触标志
    contact = next_obs[14]

    # ---------- 组件 1：货箱向泊位推进（增量形式，避免悬停收割） ----------
    # 只在"这一帧更接近泊位"时给分，且用归一化距离差
    progress = d_cur - d_nxt
    if progress < 0.0:
        progress = 0.0
    # 限制单步增量上限，防止极端跳变
    if progress > 1.0:
        progress = 1.0
    crate_to_dock_progress = 2.0 * progress

    # ---------- 组件 2：泊位内质量 shaping（仅在接近泊位时激活，门控式） ----------
    # 接近因子：d_nxt 在 [1, 3] 之间从 1 衰减到 0（容差尺度）
    if d_nxt <= 1.0:
        near_factor = 1.0
    elif d_nxt >= 3.0:
        near_factor = 0.0
    else:
        near_factor = (3.0 - d_nxt) / 2.0

    # 位置因子：越接近泊位中心越高，在容差内为 1
    if d_nxt <= 1.0:
        pos_factor = 1.0
    elif d_nxt >= 4.0:
        pos_factor = 0.0
    else:
        pos_factor = (4.0 - d_nxt) / 3.0

    # 朝向因子：角度误差 <= tol 为 1，>= 2*tol 为 0
    if ang_err <= 0.5:
        head_factor = 1.0
    elif ang_err >= 1.0:
        head_factor = 0.0
    else:
        head_factor = 1.0 - (ang_err - 0.5) / 0.5

    # 静止因子：速度 <= speed_tol 为 1，>= 4*speed_tol 为 0
    if crate_speed <= 0.05:
        still_factor = 1.0
    elif crate_speed >= 0.20:
        still_factor = 0.0
    else:
        still_factor = 1.0 - (crate_speed - 0.05) / 0.15

    # 联合质量：几何平均，避免塌缩
    quality = (pos_factor * head_factor * still_factor) ** (1.0 / 3.0)
    crate_docking_quality = 1.5 * near_factor * quality

    # ---------- 组件 3：接近泊位时的速度抑制（门控，仅接近时激活） ----------
    # 只在 near_factor > 0 时对超速部分做 hinge 惩罚
    if crate_speed > 0.05:
        excess = crate_speed - 0.05
        if excess > 0.5:
            excess = 0.5
        crate_speed_penalty_near_dock = -0.5 * near_factor * excess
    else:
        crate_speed_penalty_near_dock = 0.0

    # ---------- 组件 4：软接触惩罚（基于接触时货箱速度突变推断） ----------
    # 接触时货箱速度越大，越可能是硬碰撞
    if contact > 0.5:
        if crate_speed > 0.10:
            soft_contact_penalty = -0.3 * min(1.0, crate_speed - 0.10)
        else:
            soft_contact_penalty = 0.0
    else:
        soft_contact_penalty = 0.0

    # ---------- 组件 5：越界惩罚（hinge，仅在接近边界时生效） ----------
    # 小车位置 obs[0], obs[1] 归一化到 [-1,1]（/半宽、/半高）
    cart_x = obs[0]
    cart_y = obs[1]
    out_of_bounds_penalty = 0.0
    # 小车边界：|x| > 0.9 或 |y| > 0.9 时开始惩罚
    if cart_x > 0.9:
        out_of_bounds_penalty -= 0.5 * (cart_x - 0.9)
    if cart_x < -0.9:
        out_of_bounds_penalty -= 0.5 * (-0.9 - cart_x)
    if cart_y > 0.9:
        out_of_bounds_penalty -= 0.5 * (cart_y - 0.9)
    if cart_y < -0.9:
        out_of_bounds_penalty -= 0.5 * (-0.9 - cart_y)
    # 货箱到泊位偏移也反映货箱是否越界（泊位在仓库内）
    if abs(dx_cur) > 1.0:
        out_of_bounds_penalty -= 0.3 * (abs(dx_cur) - 1.0)
    if abs(dy_cur) > 1.0:
        out_of_bounds_penalty -= 0.3 * (abs(dy_cur) - 1.0)

    # ---------- 组件 6：动作平滑（轻量，可选约束） ----------
    action_smoothness = -0.02 * (action[0] ** 2 + action[1] ** 2)

    # ---------- 组件 7：完成事件奖励（主导信号） ----------
    # 完成条件：货箱完全在泊位内、朝向对齐、几乎静止
    inside_dock = 1.0 if (abs(dx_nxt) <= dock_tol_x and abs(dy_nxt) <= dock_tol_y) else 0.0
    aligned = 1.0 if ang_err <= 0.5 else 0.0
    still = 1.0 if crate_speed <= speed_tol else 0.0
    # 完成事件一次性大额奖励
    # 量级判据：B 必须满足 10*B > 3*(过程型单步上限之和 * 400)
    # 过程型单步上限估计：progress 2.0 + quality 1.5 + speed_penalty 0 + contact 0 + oob 0 + smooth 0.02 ≈ 3.52
    # 3 * 3.52 * 400 = 4224，10*B > 4224 => B > 422.4
    # 取 B = 500
    completion_bonus = 0.0
    if inside_dock > 0.5 and aligned > 0.5 and still > 0.5:
        completion_bonus = 500.0

    # ---------- 总奖励 ----------
    components = {
        "crate_to_dock_progress": crate_to_dock_progress,
        "crate_docking_quality": crate_docking_quality,
        "crate_speed_penalty_near_dock": crate_speed_penalty_near_dock,
        "soft_contact_penalty": soft_contact_penalty,
        "out_of_bounds_penalty": out_of_bounds_penalty,
        "action_smoothness": action_smoothness,
        "completion_bonus": completion_bonus,
    }

    total_reward = (
        crate_to_dock_progress
        + crate_docking_quality
        + crate_speed_penalty_near_dock
        + soft_contact_penalty
        + out_of_bounds_penalty
        + action_smoothness
        + completion_bonus
    )

    return (float(total_reward), components)