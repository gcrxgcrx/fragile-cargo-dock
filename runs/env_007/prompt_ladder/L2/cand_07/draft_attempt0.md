```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode 边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    components = {}

    # ---------- 几何量（货箱到泊位偏移，归一化 -> 米） ----------
    # 仓库半宽约 5.0 m，半高约 4.0 m（由卡片推断的近似恢复）
    dock_dx = next_obs[12] * 5.0
    dock_dy = next_obs[13] * 4.0
    prev_dx = obs[12] * 5.0
    prev_dy = obs[13] * 4.0

    dist_now = (dock_dx * dock_dx + dock_dy * dock_dy) ** 0.5
    dist_prev = (prev_dx * prev_dx + prev_dy * prev_dy) ** 0.5

    # 完成容差（来自环境事实）: |obs[12]| <= 0.024, |obs[13]| <= 0.030
    in_tol_x = 1.0 if abs(next_obs[12]) <= 0.024 else 0.0
    in_tol_y = 1.0 if abs(next_obs[13]) <= 0.030 else 0.0
    in_dock = 1.0 if (in_tol_x > 0.5 and in_tol_y > 0.5) else 0.0

    # 货箱速度（世界系 -> m/s）
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    # 货箱朝向误差
    crate_theta = 0.0
    crate_cos = next_obs[10]
    crate_sin = next_obs[11]
    if crate_cos >= 0.0:
        crate_theta = crate_sin
    else:
        crate_theta = 2.0 - crate_sin
    # 用更稳的方式：朝向误差近似 = |atan2(sin, cos)|，用 cos 门控
    # 这里直接用 cos 值衡量对齐：cos(误差) = crate_cos（假设泊位朝向为 0）
    align_cos = crate_cos  # 1 = 完全对齐
    aligned = 1.0 if align_cos >= 0.866 else 0.0  # 30 度

    # 完成状态判定
    done_state = 1.0 if (in_dock > 0.5 and aligned > 0.5 and crate_speed < 0.05) else 0.0

    # ---------- 1. 货箱到泊位推进（增量形式，门控对齐） ----------
    progress = dist_prev - dist_now  # 正 = 更接近
    if progress < 0.0:
        progress = 0.0
    # 对齐门控：对齐越好，推进奖励越足（避免推歪）
    align_gate = 0.5 + 0.5 * max(0.0, min(1.0, align_cos))
    # 完成状态下必须为 0
    if done_state > 0.5:
        progress_reward = 0.0
    else:
        progress_reward = 3.0 * progress * align_gate
    components["crate_to_dock_progress"] = progress_reward

    # ---------- 2. 轻柔度（接触时惩罚接近速度） ----------
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    gentleness = -2.0 * contact * closing
    components["soft_contact_penalty"] = gentleness

    # ---------- 3. 接近泊位时的速度抑制（hinge，只在接近泊位时激活） ----------
    # 距离泊位 < 0.6 m 时，货箱速度应低于 0.05 m/s
    near_dock = 1.0 if dist_now < 0.6 else 0.0
    if done_state > 0.5:
        near_speed_pen = 0.0
    else:
        excess_speed = crate_speed - 0.05
        if excess_speed < 0.0:
            excess_speed = 0.0
        near_speed_pen = -1.5 * near_dock * excess_speed
    components["dock_speed_penalty"] = near_speed_pen

    # ---------- 4. 越界守卫（小车 + 货箱） ----------
    cart_x = obs[0]
    cart_y = obs[1]
    cart_margin = 0.0
    if abs(cart_x) > 0.95:
        cart_margin = abs(cart_x) - 0.95
    if abs(cart_y) > 0.95:
        m = abs(cart_y) - 0.95
        if m > cart_margin:
            cart_margin = m
    if cart_margin < 0.0:
        cart_margin = 0.0
    cart_bound_pen = -200.0 * cart_margin

    # 货箱世界坐标恢复（近似）：cart_pos + 旋转(rel_body)
    rel_x_m = next_obs[6] * 3.0
    rel_y_m = next_obs[7] * 3.0
    ch = next_obs[2]
    sh = next_obs[3]
    crate_world_x = cart_x * 5.0 + rel_x_m * ch - rel_y_m * sh
    crate_world_y = cart_y * 4.0 + rel_x_m * sh + rel_y_m * ch
    crate_nx = crate_world_x / 5.0
    crate_ny = crate_world_y / 4.0
    crate_margin = 0.0
    if abs(crate_nx) > 0.95:
        crate_margin = abs(crate_nx) - 0.95
    if abs(crate_ny) > 0.95:
        m = abs(crate_ny) - 0.95
        if m > crate_margin:
            crate_margin = m
    if crate_margin < 0.0:
        crate_margin = 0.0
    crate_bound_pen = -200.0 * crate_margin

    components["cart_out_of_bounds_penalty"] = cart_bound_pen
    components["crate_out_of_bounds_penalty"] = crate_bound_pen

    # ---------- 5. 完成状态：其余组件必须为 0 ----------
    if done_state > 0.5:
        components["crate_to_dock_progress"] = 0.0
        components["soft_contact_penalty"] = 0.0
        components["dock_speed_penalty"] = 0.0
        # 边界惩罚保留（安全）
        # 但若已停稳在泊位内，边界惩罚应为 0（泊位在场地内）
        # 保留原值即可

    # ---------- 6. 完成事件（一次性，连续 10 步） ----------
    if done_state > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # 首次进入泊位（一次性）
    enter_event = 0.0
    if in_dock > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 20.0
    components["first_enter_dock"] = enter_event

    # 完成事件：连续 10 步满足
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["docked_success"] = success_event

    total = 0.0
    for k in components:
        total += components[k]

    return (float(total), components)
```