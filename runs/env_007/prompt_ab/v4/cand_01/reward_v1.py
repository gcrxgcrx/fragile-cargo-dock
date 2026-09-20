_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]

# 完成判据阈值（来自环境事实）
_DOCK_X_TOL = 0.024
_DOCK_Y_TOL = 0.030
_ANGLE_TOL_RAD = 0.5235987755982988  # 30 deg
_SPEED_TOL = 0.05 / 3.0              # obs 速度已除以 3.0


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 回合边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---------- 货箱到泊位度量（obs 已归一化） ----------
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    prev_dx = float(obs[12])
    prev_dy = float(obs[13])
    prev_dist = (prev_dx * prev_dx + prev_dy * prev_dy) ** 0.5

    # ---------- 货箱朝向误差 ----------
    ccos = float(next_obs[10])
    csin = float(next_obs[11])
    ang_err = abs(ccos * 0.0 + csin * 0.0)  # placeholder 防未定义
    # 用 atan2 形式：误差 = |atan2(sin,cos)|，cos 为 1 时误差 0
    # 避免 import，用近似：err_cos = 1 - cos(theta)
    ang_err = 1.0 - ccos  # 0 (对齐) ~ 2 (反向)，单调

    # ---------- 货箱速度 ----------
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---------- 是否处于完成状态 ----------
    in_dock = (abs(dx) <= _DOCK_X_TOL) and (abs(dy) <= _DOCK_Y_TOL)
    aligned = ang_err < (1.0 - 0.8660254037844387)  # cos(30deg)≈0.866
    slow = crate_speed < 0.05
    completed_state = in_dock and aligned and slow

    # ---------- 连续计数 ----------
    if completed_state:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    components = {}

    # ---------- 一次性完成事件 ----------
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["success_event"] = success_event

    # ---------- 首次进入泊位（一次性） ----------
    enter_event = 0.0
    if in_dock and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 20.0
    components["enter_dock_event"] = enter_event

    # ---------- 完成态：除一次性事件外全部为 0 ----------
    if completed_state:
        total = success_event + enter_event
        return (float(total), components)

    # ---------- 主进展：货箱到泊位的增量改善（门控对齐） ----------
    improvement = prev_dist - dist
    if improvement < 0.0:
        improvement = 0.0
    # 对齐门控：越对齐越给分（只在推进时生效）
    align_gate = 1.0 - 0.5 * ang_err
    if align_gate < 0.0:
        align_gate = 0.0
    progress_reward = 30.0 * improvement * align_gate
    components["crate_to_dock_progress"] = progress_reward

    # ---------- 接触轻柔度（唯一教减速的信号） ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing
    components["soft_contact"] = gentleness

    # ---------- 接近泊位时的速度抑制（hinge，仅惩罚过快） ----------
    speed_pen = 0.0
    if dist < 0.15:
        excess = crate_speed - 0.15
        if excess > 0.0:
            speed_pen = -2.0 * excess
    components["speed_near_dock"] = speed_pen

    # ---------- 越界 hinge 惩罚 ----------
    oob = 0.0
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    if abs(cart_x) > 0.95:
        oob -= 5.0 * (abs(cart_x) - 0.95)
    if abs(cart_y) > 0.95:
        oob -= 5.0 * (abs(cart_y) - 0.95)
    # 货箱越界近似：货箱世界坐标恢复
    rx = float(next_obs[6]) * 3.0
    ry = float(next_obs[7]) * 3.0
    ch = float(next_obs[2])
    sh = float(next_obs[3])
    wx = cart_x * 5.0 + rx * ch - ry * sh
    wy = cart_y * 4.0 + rx * sh + ry * ch
    if abs(wx) > 4.75:
        oob -= 5.0 * (abs(wx) - 4.75)
    if abs(wy) > 3.8:
        oob -= 5.0 * (abs(wy) - 3.8)
    components["out_of_bounds"] = oob

    total = (
        components["success_event"]
        + components["enter_dock_event"]
        + components["crate_to_dock_progress"]
        + components["soft_contact"]
        + components["speed_near_dock"]
        + components["out_of_bounds"]
    )
    return (float(total), components)