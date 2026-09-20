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

    # ---------- 几何量（泊位容差：|dx|<=0.024, |dy|<=0.030）----------
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    odx = float(obs[12])
    ody = float(obs[13])

    # 归一化距离度量（用容差尺度归一，避免悬停陷阱）
    dist = (dx * dx + dy * dy) ** 0.5
    odist = (odx * odx + ody * ody) ** 0.5

    # ---------- 1) 主进度：货箱向泊位靠近的增量 ----------
    progress = odist - dist
    if progress > 0.05:
        progress = 0.05
    if progress < -0.05:
        progress = -0.05
    components["crate_to_dock_progress"] = 12.0 * progress

    # ---------- 2) 货箱朝向对齐 shaping（接近泊位时生效的门控）----------
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    align = crate_cos  # cos(朝向误差)，1 表示对齐
    if align < 0.0:
        align = 0.0
    # 只在接近泊位时给对齐引导，避免全局收分
    near_gate = 1.0 / (1.0 + 8.0 * dist)
    components["crate_align"] = 0.6 * align * near_gate

    # ---------- 3) 轻柔接触：接触中且接近时的惩罚 ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing
    components["soft_contact"] = gentleness

    # ---------- 4) 接近泊位时的速度抑制（门控，仅近距离生效）----------
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    speed_gate = 1.0 / (1.0 + 25.0 * dist)
    components["dock_speed_damp"] = -0.4 * speed_gate * crate_speed

    # ---------- 5) 边界 hinge（小车越界风险）----------
    cx = abs(float(next_obs[0]))
    cy = abs(float(next_obs[1]))
    oob = 0.0
    if cx > 0.92:
        oob += (cx - 0.92)
    if cy > 0.92:
        oob += (cy - 0.92)
    components["cart_oob"] = -3.0 * oob

    # 货箱越界风险（用货箱到泊位偏移 + 泊位位置无法直接得，用相对量近似）
    crate_oob = 0.0
    if abs(dx) > 1.2:
        crate_oob += (abs(dx) - 1.2)
    if abs(dy) > 1.2:
        crate_oob += (abs(dy) - 1.2)
    components["crate_oob"] = -3.0 * crate_oob

    # ---------- 6) 动作平滑（轻量）----------
    a0 = float(action[0])
    a1 = float(action[1])
    components["action_smooth"] = -0.02 * (a0 * a0 + a1 * a1)

    # ---------- 完成条件（显式从 obs 推断）----------
    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0
    aligned = 1.0 if crate_cos >= 0.866 else 0.0  # 朝向误差 < 30°
    still = 1.0 if crate_speed < 0.05 else 0.0

    if inside > 0.5 and aligned > 0.5 and still > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # 首次进入泊位：一次性奖励
    enter_event = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 8.0
    components["first_enter_dock"] = enter_event

    # 完成事件：一次性，仅发一次
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["docked_success"] = success_event

    total = 0.0
    for k in components:
        total += components[k]

    return float(total), components