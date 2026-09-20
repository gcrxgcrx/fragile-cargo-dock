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
    # 泊位偏移（归一化到半宽/半高）
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    # 真实米制偏移：半宽 5.0? 事实未给米制半宽，用归一化度量即可
    dist = (dx * dx + dy * dy) ** 0.5
    prev_dx = float(obs[12])
    prev_dy = float(obs[13])
    prev_dist = (prev_dx * prev_dx + prev_dy * prev_dy) ** 0.5

    # 完成判据（只能用环境给定容差）
    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0

    # 货箱速度（世界系，恢复为 m/s）
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 朝向误差
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    heading_err = abs(ch - 1.0) + abs(sh)  # cos 越接近1、sin越接近0越好
    aligned = 1.0 if heading_err < 0.134 else 0.0  # cos(30°)=0.866 -> err~0.134

    # ---------- 组件 1：货箱向泊位推进（增量形式，避免悬停收割） ----------
    progress = prev_dist - dist
    if progress > 0.0:
        w_prog = 2.0
    else:
        w_prog = 0.5
    crate_to_dock_progress = w_prog * progress

    # ---------- 组件 2：接近泊位时的静止/对齐联合引导（仅在接近时激活） ----------
    near_gate = 1.0 if dist < 0.15 else 0.0
    speed_term = 0.0
    if near_gate > 0.0:
        # 速度越低越好，用 hinge 形式：低于阈值给正，高于阈值给负
        speed_term = 0.3 * max(0.0, 1.0 - crate_speed / 0.5)
        # 对齐引导
        align_term = 0.2 * max(0.0, 1.0 - heading_err / 0.5)
    else:
        align_term = 0.0
    docking_quality = speed_term + align_term

    # ---------- 组件 3：接触轻柔度（唯一教会减速的信号） ----------
    cart_vx = float(next_obs[2])
    cart_vy = float(next_obs[3])
    crate_along_heading = cvx * cart_vx + cvy * cart_vy
    closing = float(next_obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---------- 组件 4：越界防护（hinge，仅在接近边界时） ----------
    # 小车位置归一化
    cx = float(next_obs[0])
    cy = float(next_obs[1])
    bound_pen = 0.0
    margin = 0.85
    if abs(cx) > margin:
        bound_pen -= 0.5 * (abs(cx) - margin)
    if abs(cy) > margin:
        bound_pen -= 0.5 * (abs(cy) - margin)

    # ---------- 组件 5：动作平滑（轻量） ----------
    smooth = -0.01 * (float(action[0]) ** 2 + float(action[1]) ** 2)

    # ---------- 完成事件：连续 10 步满足条件 ----------
    if inside > 0.5 and aligned > 0.5 and crate_speed < 0.05:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # ---------- 首次进入泊位一次性奖励 ----------
    enter_bonus = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_bonus = 20.0

    components = {
        "crate_to_dock_progress": crate_to_dock_progress,
        "docking_quality": docking_quality,
        "gentleness": gentleness,
        "bound_penalty": bound_pen,
        "action_smooth": smooth,
        "enter_bonus": enter_bonus,
        "success_event": success_event,
    }

    total = (crate_to_dock_progress + docking_quality + gentleness
             + bound_pen + smooth + enter_bonus + success_event)
    return float(total), components