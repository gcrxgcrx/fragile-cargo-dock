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

    # ---------- 几何量 ----------
    # 泊位容差（环境事实给出）：|obs[12]| <= 0.024, |obs[13]| <= 0.030
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    odx = float(obs[12])
    ody = float(obs[13])

    # 归一化距离（用容差尺度归一，便于比较）
    dist = ((dx / 0.024) ** 2 + (dy / 0.030) ** 2) ** 0.5
    odist = ((odx / 0.024) ** 2 + (ody / 0.030) ** 2) ** 0.5

    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0

    # 朝向误差（货箱朝向，rad）
    ang_err = abs(float(next_obs[11]))  # sin 分量近似小角误差
    align = max(0.0, 1.0 - ang_err / 0.5236)  # 30° = 0.5236 rad

    # 货箱速度（m/s）
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    slow = max(0.0, 1.0 - crate_speed / 0.05)

    # ---------- 完成条件（显式推断）----------
    done_cond = (inside > 0.5) and (ang_err < 0.5236) and (crate_speed < 0.05)

    if done_cond:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- 组件 ----------
    components = {}

    # 1) 主进度：增量形式（只在更接近泊位时给分）
    progress = 0.0
    if odist > dist:
        progress = 0.5 * (odist - dist)
    components["crate_to_dock_progress"] = progress

    # 2) 朝向对齐 × 靠近增量（门控，不单独按步给分）
    align_progress = 0.0
    if odist > dist:
        align_progress = 0.2 * align * (odist - dist)
    components["align_gated_progress"] = align_progress

    # 3) 接触轻柔度（唯一能教会减速的信号）
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing
    components["soft_contact_penalty"] = gentleness

    # 4) 接近泊位时抑制货箱速度（hinge：只在接近泊位且过快时惩罚）
    near_dock = 1.0 if dist < 3.0 else 0.0
    speed_pen = 0.0
    if near_dock > 0.5 and crate_speed > 0.05:
        speed_pen = -0.3 * (crate_speed - 0.05)
    components["crate_speed_penalty_near_dock"] = speed_pen

    # 5) 越界惩罚（hinge）
    oob = 0.0
    cx = float(next_obs[0])
    cy = float(next_obs[1])
    if abs(cx) > 0.95:
        oob -= 0.5 * (abs(cx) - 0.95)
    if abs(cy) > 0.95:
        oob -= 0.5 * (abs(cy) - 0.95)
    components["out_of_bounds_penalty"] = oob

    # 6) 障碍接近惩罚（前方）
    front = float(next_obs[15])
    obs_pen = 0.0
    if front > 0.7:
        obs_pen = -0.2 * (front - 0.7)
    components["obstacle_penalty"] = obs_pen

    # 7) 首次进入泊位（一次性）
    enter_bonus = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_bonus = 20.0
    components["first_enter_dock"] = enter_bonus

    # 8) 完成事件（一次性，连续 10 步）
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["docked_success"] = success_event

    # ---------- 完成状态下，除一次性事件外必须恰好为 0 ----------
    if done_cond and _ENTERED[0]:
        # 除了一次性事件，其余过程项清零（已完成状态不再收分）
        components["crate_to_dock_progress"] = 0.0
        components["align_gated_progress"] = 0.0
        components["soft_contact_penalty"] = 0.0
        components["crate_speed_penalty_near_dock"] = 0.0
        components["out_of_bounds_penalty"] = 0.0
        components["obstacle_penalty"] = 0.0
        # first_enter_dock 和 docked_success 保留（一次性）

    total = 0.0
    for k in components:
        total += components[k]

    return (float(total), components)