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

    # ---------- 几何 / 运动学量 ----------
    # 泊位容差（来自环境事实）
    dx = float(next_obs[12])   # 货箱中心到泊位中心 x 偏移 / 半宽
    dy = float(next_obs[13])   # 货箱中心到泊位中心 y 偏移 / 半高
    in_dock = (abs(dx) <= 0.024) and (abs(dy) <= 0.030)

    # 货箱速度（世界系，m/s）
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    slow = crate_speed < 0.05

    # 货箱朝向误差（rad）
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    heading_err = abs((sh * sh) ** 0.5)  # 占位，下面用 atan2 形式
    # 用 cos 值判断对齐：cos(err) = ch（假设 cos/sin 归一）
    cos_err = ch
    if cos_err > 1.0:
        cos_err = 1.0
    if cos_err < -1.0:
        cos_err = -1.0
    aligned = cos_err > 0.8660254  # 30°

    # ---------- 完成条件 ----------
    done_cond = in_dock and aligned and slow
    if done_cond:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # 完成状态：除一次性事件外所有组件必须为 0
    if done_cond:
        success_event = 0.0
        if _STREAK[0] >= 10 and not _PAID[0]:
            _PAID[0] = True
            success_event = 300.0
        components = {
            "success_event": success_event,
            "crate_progress": 0.0,
            "align_gated_progress": 0.0,
            "gentleness": 0.0,
            "speed_near_dock_penalty": 0.0,
            "out_of_bounds_penalty": 0.0,
        }
        return (float(success_event), components)

    # ---------- 主进展信号：货箱到泊位距离的增量 ----------
    # 用归一化偏移的欧氏距离作为度量（近似，容差尺度一致）
    prev_d = (float(obs[12]) ** 2 + float(obs[13]) ** 2) ** 0.5
    curr_d = (dx * dx + dy * dy) ** 0.5
    progress = prev_d - curr_d  # >0 表示更接近
    if progress < -0.5:
        progress = -0.5
    if progress > 0.5:
        progress = 0.5

    # 对齐门控：对齐越好，进展信号越有效（不单独给分）
    align_gate = (cos_err + 1.0) * 0.5  # 0..1
    crate_progress = 6.0 * progress
    align_gated_progress = 2.0 * progress * align_gate

    # ---------- 轻柔度（接触中且接近才惩罚） ----------
    cart_vx = float(next_obs[2])
    cart_vy = float(next_obs[3])
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * cart_vx + crate_vy * cart_vy
    closing = float(next_obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---------- 接近泊位时的高速惩罚（hinge，仅靠近时激活） ----------
    # 距离泊位越近，速度惩罚越强；只在未完成时
    near_dock = 1.0 - min(1.0, curr_d / 0.5)  # curr_d<0.5 时 >0
    if near_dock < 0.0:
        near_dock = 0.0
    speed_excess = crate_speed - 0.05
    if speed_excess < 0.0:
        speed_excess = 0.0
    speed_near_dock_penalty = -1.0 * near_dock * speed_excess

    # ---------- 越界惩罚（hinge，接近 ±1 归一化边界） ----------
    oob = 0.0
    cx = float(next_obs[0])
    cy = float(next_obs[1])
    for v in (cx, cy):
        if v > 0.9:
            oob += (v - 0.9)
        if v < -0.9:
            oob += (-0.9 - v)
    out_of_bounds_penalty = -3.0 * oob

    # ---------- 首次进入泊位（一次性） ----------
    enter_event = 0.0
    if in_dock and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 10.0

    components = {
        "crate_progress": float(crate_progress),
        "align_gated_progress": float(align_gated_progress),
        "gentleness": float(gentleness),
        "speed_near_dock_penalty": float(speed_near_dock_penalty),
        "out_of_bounds_penalty": float(out_of_bounds_penalty),
        "enter_event": float(enter_event),
        "success_event": 0.0,
    }
    total = (
        crate_progress
        + align_gated_progress
        + gentleness
        + speed_near_dock_penalty
        + out_of_bounds_penalty
        + enter_event
    )
    return (float(total), components)