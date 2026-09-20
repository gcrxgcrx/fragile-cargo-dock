_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode 边界检测（obs[18] 单调递增，重置时回落） ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---------- 泊位几何（半宽 5.0 m，半高 4.0 m；容差 0.12 m / 0.12 m） ----------
    ox = float(next_obs[12])   # 归一化 x 偏移
    oy = float(next_obs[13])   # 归一化 y 偏移
    dist_norm = (ox * ox + oy * oy) ** 0.5
    dist_m = dist_norm * 5.0

    # 完全进入泊位：|ox|<=0.024, |oy|<=0.030
    inside = 1.0 if (abs(ox) <= 0.024 and abs(oy) <= 0.030) else 0.0

    # ---------- 朝向对齐（货箱朝向误差 < 30°） ----------
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    nrm = (ch * ch + sh * sh) ** 0.5
    if nrm < 1e-6:
        nrm = 1e-6
    ch = ch / nrm
    sh = sh / nrm
    if ch < 0.0:
        ch = -ch
        sh = -sh
    align_err = (sh * sh) ** 0.5          # |sin(theta)|
    aligned = 1.0 if align_err < 0.5 else 0.0   # 30° -> sin=0.5

    # ---------- 货箱速度 ----------
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    cspeed = (cvx * cvx + cvy * cvy) ** 0.5
    slow = 1.0 if cspeed < 0.05 else 0.0

    # ---------- 组件 1：货箱到泊位的增量进展 ----------
    ox0 = float(obs[12])
    oy0 = float(obs[13])
    dist_prev = (ox0 * ox0 + oy0 * oy0) ** 0.5
    progress = (dist_prev - dist_norm) * 5.0
    if progress > 1.0:
        progress = 1.0
    if progress < -1.0:
        progress = -1.0
    r_progress = 12.0 * progress

    # ---------- 组件 2：接触轻柔度（唯一能教"接近泊位时减速"的信号） ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    r_gentle = -0.05 * contact * closing

    # ---------- 组件 3：接近泊位时的速度抑制（门控，不惩罚正常匀速推箱） ----------
    gate_near = max(0.0, 1.0 - dist_norm / 0.10)
    if gate_near > 1.0:
        gate_near = 1.0
    r_speed_near = -0.30 * gate_near * cspeed

    # ---------- 组件 4：对齐 shaping（仅在接近泊位时激活） ----------
    r_align = 0.30 * gate_near * (1.0 - align_err)

    # ---------- 组件 5：静止 shaping（仅在已完全进入泊位时激活） ----------
    r_still = 0.50 * inside * max(0.0, 1.0 - cspeed / 0.05)

    # ---------- 组件 6：边界 hinge 惩罚 ----------
    r_bounds = 0.0
    cx = abs(float(next_obs[0]))
    cy = abs(float(next_obs[1]))
    if cx > 0.90:
        r_bounds -= 1.0 * (cx - 0.90)
    if cy > 0.90:
        r_bounds -= 1.0 * (cy - 0.90)
    if abs(ox) > 0.95:
        r_bounds -= 1.0 * (abs(ox) - 0.95)
    if abs(oy) > 0.95:
        r_bounds -= 1.0 * (abs(oy) - 0.95)

    # ---------- 组件 7：动作平滑（轻量） ----------
    r_smooth = -0.01 * (float(action[0]) ** 2 + float(action[1]) ** 2)

    # ---------- 完成条件计数 ----------
    done_cond = (inside > 0.5) and (aligned > 0.5) and (slow > 0.5)
    if done_cond:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- 一次性事件：首次进入泊位 / 完成 ----------
    r_enter = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        r_enter = 50.0

    r_success = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        r_success = 300.0

    total = (
        r_progress
        + r_gentle
        + r_speed_near
        + r_align
        + r_still
        + r_bounds
        + r_smooth
        + r_enter
        + r_success
    )

    components = {
        "crate_to_dock_progress": r_progress,
        "soft_contact_gentleness": r_gentle,
        "crate_speed_near_dock": r_speed_near,
        "crate_align_near_dock": r_align,
        "crate_still_in_dock": r_still,
        "out_of_bounds_penalty": r_bounds,
        "action_smoothness": r_smooth,
        "first_entered_dock": r_enter,
        "docking_success_event": r_success,
    }

    return float(total), components