_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]

_DOCK_X_TOL = 0.024
_DOCK_Y_TOL = 0.030
_SPEED_TOL = 0.05
_ALIGN_COS_TOL = 0.8660254  # cos(30 deg)


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 回合边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---------- 基础量 ----------
    # 货箱到泊位的有符号偏移（归一化）
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    pdx = float(obs[12])
    pdy = float(obs[13])

    # 货箱速度（世界系，m/s）
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向误差（弧度）
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    # 泊位朝向假定与仓库轴对齐（朝 +x），误差角 = atan2(sin, cos)
    align_cos = crate_cos  # 与 0 度朝向的余弦对齐度
    if align_cos > 1.0:
        align_cos = 1.0
    if align_cos < -1.0:
        align_cos = -1.0

    # 接触
    contact = 1.0 if next_obs[14] > 0.5 else 0.0

    # 车头方向
    hx = float(obs[2])
    hy = float(obs[3])

    # ---------- 完成条件判定 ----------
    inside = (abs(dx) <= _DOCK_X_TOL) and (abs(dy) <= _DOCK_Y_TOL)
    aligned = align_cos >= _ALIGN_COS_TOL
    slow = crate_speed < _SPEED_TOL
    done_state = inside and aligned and slow

    if done_state:
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
    if inside and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 20.0
    components["enter_dock_bonus"] = enter_event

    if done_state:
        # 完成状态下，除一次性事件外其余组件恰好为 0
        total = success_event + enter_event
        return (float(total), components)

    # ---------- 主进度信号：货箱到泊位距离的增量（米制近似） ----------
    # 归一化偏移量按半宽/半高换算（半宽 5.0, 半高 4.0 近似）
    prev_dist = ((pdx * 5.0) ** 2 + (pdy * 4.0) ** 2) ** 0.5
    curr_dist = ((dx * 5.0) ** 2 + (dy * 4.0) ** 2) ** 0.5
    delta_dist = prev_dist - curr_dist  # 正 = 更接近

    # 门控：对齐度越高、越接近泊位，进度信号权重越大
    align_gate = 0.5 + 0.5 * max(0.0, align_cos)
    progress = 3.0 * delta_dist * align_gate
    components["crate_to_dock_progress"] = progress

    # ---------- 接触轻柔度（接近速度惩罚） ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * hx + crate_vy * hy
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    gentleness = -0.05 * contact * closing
    components["soft_contact"] = gentleness

    # ---------- 接近泊位时的速度抑制（hinge，仅在接近泊位时启用） ----------
    near_dock = 1.0 / (1.0 + 3.0 * curr_dist)
    speed_excess = crate_speed - 0.3
    if speed_excess < 0.0:
        speed_excess = 0.0
    speed_penalty = -0.5 * near_dock * (speed_excess ** 2)
    components["near_dock_speed_penalty"] = speed_penalty

    # ---------- 边界惩罚（小车与货箱） ----------
    bx = abs(float(next_obs[0]))
    by = abs(float(next_obs[1]))
    cart_margin = 0.9
    cart_excess_x = bx - cart_margin
    if cart_excess_x < 0.0:
        cart_excess_x = 0.0
    cart_excess_y = by - cart_margin
    if cart_excess_y < 0.0:
        cart_excess_y = 0.0
    # 货箱世界坐标近似恢复
    crate_wx = float(next_obs[0]) * 5.0 + float(next_obs[6]) * 3.0 * hx - float(next_obs[7]) * 3.0 * hy
    crate_wy = float(next_obs[1]) * 4.0 + float(next_obs[6]) * 3.0 * hy + float(next_obs[7]) * 3.0 * hx
    crate_bx = abs(crate_wx) / 5.0
    crate_by = abs(crate_wy) / 4.0
    crate_excess_x = crate_bx - 0.9
    if crate_excess_x < 0.0:
        crate_excess_x = 0.0
    crate_excess_y = crate_by - 0.9
    if crate_excess_y < 0.0:
        crate_excess_y = 0.0
    oob = -2.0 * (cart_excess_x ** 2 + cart_excess_y ** 2 + crate_excess_x ** 2 + crate_excess_y ** 2)
    components["out_of_bounds"] = oob

    # ---------- 动作平滑（轻量，防止抖动） ----------
    smooth = -0.01 * (float(action[0]) ** 2 + float(action[1]) ** 2)
    components["action_smoothness"] = smooth

    total = 0.0
    for k in components:
        total += components[k]
    return (float(total), components)