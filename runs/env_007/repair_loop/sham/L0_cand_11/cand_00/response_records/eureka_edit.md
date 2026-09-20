# Response Record

分析：训练反馈中的组件数值出现明显错位（mean/abs_mean/min/max 列互相串位），但可确认的关键事实是：20/20 episode 全部 truncation、native score 为负、episode 长度恒为 400，说明策略从未成功停靠，主信号（progress + proximity）量级远小于各惩罚项，且 `near_dock_gate` 式的速度抑制在接近泊位时惩罚了"推进所必需的运动"，与"不得对推进任务产生净负贡献"冲突。此外 `crate_docking_quality` 用 `(crate_cos+1)/2` 对齐因子在货箱朝向为 ±π 时也判为对齐，语义错误。改进方向：把速度抑制改为"仅在货箱已进入泊位且朝向对齐时才生效"的门控惩罚（不作为全局持续扣分），把对齐因子改为真正的角度误差 bounded 形式，凸化主进度信号并让联合完成代理主导，弱化各惩罚权重，加入进入泊位的稀疏 bonus 以打破 truncation 僵局。

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    DOCK_X_TOL = 0.024
    DOCK_Y_TOL = 0.030
    ANGLE_TOL = 0.5235987755982988
    SPEED_TOL = 0.05 / 3.0

    cur_dx = obs[12]
    cur_dy = obs[13]
    nxt_dx = next_obs[12]
    nxt_dy = next_obs[13]

    cur_dist = (cur_dx * cur_dx + cur_dy * cur_dy) ** 0.5
    nxt_dist = (nxt_dx * nxt_dx + nxt_dy * nxt_dy) ** 0.5

    # ---- 主进度信号：凸化 improvement delta ----
    raw_progress = (cur_dist - nxt_dist)
    progress = raw_progress * 12.0
    if raw_progress > 0.0:
        progress = progress * (1.0 + 1.5 * raw_progress)

    # ---- 泊位接近度：凸化，越近梯度越强 ----
    proximity = 1.0 / (1.0 + 5.0 * nxt_dist)
    proximity = proximity * proximity

    # ---- 货箱朝向误差（真实角度）----
    crate_cos = next_obs[10]
    crate_sin = next_obs[11]
    crate_angle_err = (crate_sin * crate_sin + (1.0 - crate_cos) * (1.0 - crate_cos)) ** 0.5
    if crate_angle_err > 3.141592653589793:
        crate_angle_err = 3.141592653589793
    align_factor = max(0.0, 1.0 - crate_angle_err / 1.5707963267948966)

    # ---- 货箱速度 ----
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    crate_speed_norm = crate_speed / 3.0

    # ---- 是否已进入泊位（几何重建）----
    in_dock = 1.0 if (abs(nxt_dx) <= DOCK_X_TOL and abs(nxt_dy) <= DOCK_Y_TOL) else 0.0

    # ---- 速度抑制：仅在已进入泊位且朝向对齐时门控生效 ----
    align_gate = 1.0 if crate_angle_err < ANGLE_TOL else 0.0
    speed_penalty = -0.6 * in_dock * align_gate * (crate_speed_norm ** 2)

    # ---- 联合完成代理：进入 + 对齐 + 静止 ----
    pos_factor = 1.0 / (1.0 + 25.0 * nxt_dist)
    still_factor = 1.0 / (1.0 + 30.0 * crate_speed_norm)
    joint_proxy = (pos_factor * align_factor * still_factor) ** (1.0 / 3.0)

    # ---- 进入泊位稀疏 bonus（打破 truncation 僵局）----
    dock_entry_bonus = 2.0 * in_dock

    # ---- 轻柔接触：仅在接触且速度突变大时 hinge ----
    contact = next_obs[14]
    prev_crate_vx = obs[8] * 3.0
    prev_crate_vy = obs[9] * 3.0
    dvx = crate_vx - prev_crate_vx
    dvy = crate_vy - prev_crate_vy
    dv_mag = (dvx * dvx + dvy * dvy) ** 0.5
    soft_contact_penalty = 0.0
    if contact > 0.5:
        excess = dv_mag - 1.5
        if excess > 0.0:
            soft_contact_penalty = -0.15 * excess

    # ---- 越界惩罚（hinge）----
    cart_x = obs[0]
    cart_y = obs[1]
    out_penalty = 0.0
    cart_excess_x = abs(cart_x) - 0.92
    if cart_excess_x > 0.0:
        out_penalty -= 0.8 * cart_excess_x
    cart_excess_y = abs(cart_y) - 0.92
    if cart_excess_y > 0.0:
        out_penalty -= 0.8 * cart_excess_y
    crate_excess = nxt_dist - 1.6
    if crate_excess > 0.0:
        out_penalty -= 0.8 * crate_excess

    # ---- 障碍接近惩罚（hinge）----
    obs_penalty = 0.0
    front = next_obs[15]
    left = next_obs[16]
    right = next_obs[17]
    if front > 0.9:
        obs_penalty -= 0.15 * (front - 0.9)
    if left > 0.9:
        obs_penalty -= 0.15 * (left - 0.9)
    if right > 0.9:
        obs_penalty -= 0.15 * (right - 0.9)

    # ---- 动作平滑（极轻量）----
    drive = action[0]
    steer = action[1]
    action_penalty = -0.01 * (drive * drive + steer * steer)

    comp_dock_progress = progress * 1.0
    comp_dock_proximity = proximity * 2.0
    comp_dock_quality = joint_proxy * 3.0
    comp_dock_entry = dock_entry_bonus * 1.0
    comp_speed_penalty = speed_penalty
    comp_soft_contact = soft_contact_penalty
    comp_out_of_bounds = out_penalty
    comp_obstacle = obs_penalty
    comp_action_smooth = action_penalty

    total = (
        comp_dock_progress
        + comp_dock_proximity
        + comp_dock_quality
        + comp_dock_entry
        + comp_speed_penalty
        + comp_soft_contact
        + comp_out_of_bounds
        + comp_obstacle
        + comp_action_smooth
    )

    components = {
        "crate_to_dock_progress": float(comp_dock_progress),
        "crate_dock_proximity": float(comp_dock_proximity),
        "crate_docking_quality": float(comp_dock_quality),
        "crate_dock_entry_bonus": float(comp_dock_entry),
        "crate_speed_penalty_near_dock": float(comp_speed_penalty),
        "soft_contact_penalty": float(comp_soft_contact),
        "out_of_bounds_penalty": float(comp_out_of_bounds),
        "obstacle_proximity_penalty": float(comp_obstacle),
        "action_smoothness_penalty": float(comp_action_smooth),
    }

    return float(total), components
