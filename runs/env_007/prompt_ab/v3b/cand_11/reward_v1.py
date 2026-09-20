_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_DIST = [None]

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode 边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = None
    _PREV_T[0] = t

    components = {}

    # ---------- 泊位几何（环境事实给出） ----------
    # 货箱中心到泊位中心的有符号偏移（归一化）
    dx_prev = float(obs[12])
    dy_prev = float(obs[13])
    dx_next = float(next_obs[12])
    dy_next = float(next_obs[13])

    # 归一化距离（用于增量与门控）
    dist_prev = (dx_prev * dx_prev + dy_prev * dy_prev) ** 0.5
    dist_next = (dx_next * dx_next + dy_next * dy_next) ** 0.5

    # 容差（环境事实：|obs[12]| <= 0.024 且 |obs[13]| <= 0.030）
    inside = 1.0 if (abs(dx_next) <= 0.024 and abs(dy_next) <= 0.030) else 0.0

    # 接近泊位的门控（0=远, 1=很近），用于只在近处激活速度抑制
    near_gate = max(0.0, min(1.0, (0.35 - dist_next) / 0.25))
    if near_gate < 0.0:
        near_gate = 0.0

    # ---------- 1) 主进度：货箱向泊位的增量靠近 ----------
    progress = 0.0
    if _PREV_DIST[0] is None:
        _PREV_DIST[0] = dist_prev
    delta = _PREV_DIST[0] - dist_next
    if delta > 0.0:
        progress = 50.0 * delta
    _PREV_DIST[0] = dist_next
    components["crate_to_dock_progress"] = progress

    # ---------- 2) 泊位内质量 shaping（仅当已在容差内才给，门控式） ----------
    # 朝向对齐：货箱朝向误差 < 30°
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    # 朝向误差的余弦（1 = 对齐）
    align = crate_cos
    if align < 0.0:
        align = 0.0
    align_factor = max(0.0, min(1.0, (align - 0.8660) / 0.1340))  # cos30°≈0.866

    # 静止因子：货箱速度 < 0.05 m/s
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    still_factor = max(0.0, min(1.0, (0.20 - crate_speed) / 0.15))

    dock_quality = 0.0
    if inside > 0.5:
        dock_quality = 3.0 * align_factor * still_factor
    components["crate_docking_quality"] = dock_quality

    # ---------- 3) 轻柔度：接触且接近时惩罚 ----------
    crate_along_heading = cvx * float(obs[2]) + cvy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing
    components["soft_contact"] = gentleness

    # ---------- 4) 近泊位速度抑制（门控式，仅近处生效） ----------
    # 只在接近泊位且货箱仍有速度时抑制，不惩罚正常远距离推动
    speed_near_dock = 0.0
    if near_gate > 0.0:
        overspeed = crate_speed - 0.15
        if overspeed < 0.0:
            overspeed = 0.0
        speed_near_dock = -1.5 * near_gate * overspeed
    components["speed_near_dock"] = speed_near_dock

    # ---------- 5) 越界软惩罚（hinge，仅在接近边界时生效） ----------
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    oob = 0.0
    if abs(cart_x) > 0.90:
        oob -= 2.0 * (abs(cart_x) - 0.90)
    if abs(cart_y) > 0.90:
        oob -= 2.0 * (abs(cart_y) - 0.90)
    components["out_of_bounds"] = oob

    # ---------- 6) 动作平滑（轻量） ----------
    a0 = float(action[0])
    a1 = float(action[1])
    components["action_smoothness"] = -0.01 * (a0 * a0 + a1 * a1)

    # ---------- 7) 完成事件（一次性） ----------
    # 完成条件：货箱完全在泊位内 + 朝向误差 < 30° + 速度 < 0.05 m/s
    if inside > 0.5 and align_factor > 0.99 and crate_speed < 0.05:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["success_event"] = success_event

    # ---------- 8) 首次进入泊位（一次性） ----------
    first_enter = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        first_enter = 30.0
    components["first_enter_dock"] = first_enter

    total = 0.0
    for k in components:
        total += components[k]

    return float(total), components