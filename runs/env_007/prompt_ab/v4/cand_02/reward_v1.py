_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 回合边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---------- 几何量（货箱到泊位偏移，已归一化） ----------
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    pdx = float(obs[12])
    pdy = float(obs[13])
    prev_dist = (pdx * pdx + pdy * pdy) ** 0.5

    # 泊位容差（来自环境事实）：|dx|<=0.024, |dy|<=0.030
    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0

    # ---------- 货箱速度 ----------
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---------- 朝向误差 ----------
    cc = float(next_obs[10])
    cs = float(next_obs[11])
    # 泊位对齐方向：货箱朝向与目标朝向的角误差（用 cos 近似 1 为对齐）
    # 用 atan2 误差的余弦绝对值形式：|cos(theta_err)| ~ 1 表示对齐
    # 这里直接使用货箱朝向与 x 轴对齐（泊位朝向）的偏差
    heading_align = cc  # cos(heading)，|heading_align| 接近 1 表示轴向对齐
    if heading_align < 0.0:
        heading_align = -heading_align  # 允许 180° 对称
    # 对齐门：cos(30°) ≈ 0.866
    align_gate = 1.0 if heading_align >= 0.866 else 0.0

    # ---------- 完成条件 ----------
    slow = 1.0 if crate_speed < 0.05 else 0.0
    if inside > 0.5 and align_gate > 0.5 and slow > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- 组件 ----------
    components = {}

    # 1) 主推进：货箱到泊位的距离增量（增量形式，避免悬停收分）
    delta = prev_dist - dist
    if delta < 0.0:
        delta = 0.0
    # 对齐门控：只有朝向大致对齐时，靠近增量才给正分
    # 但即使未对齐，也允许小比例给分以引导探索
    progress = 3.0 * delta * (0.3 + 0.7 * heading_align)
    components["crate_progress"] = progress

    # 2) 轻柔接触惩罚（核心技能信号）
    crate_along_heading = cvx * float(obs[2]) + cvy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing
    components["soft_contact"] = gentleness

    # 3) 接近泊位时的速度抑制（仅当接近泊位才启用，且只惩罚不奖）
    # 用 hinge：距离泊位较近且速度高时惩罚
    if dist < 0.10:
        near_factor = 1.0 - dist / 0.10
        if near_factor < 0.0:
            near_factor = 0.0
        speed_excess = crate_speed - 0.10
        if speed_excess < 0.0:
            speed_excess = 0.0
        components["dock_speed_penalty"] = -0.5 * near_factor * (speed_excess ** 2)
    else:
        components["dock_speed_penalty"] = 0.0

    # 4) 越界惩罚（小车 + 货箱估计）
    # 小车位置 obs[0], obs[1] 归一化到仓库半宽/半高，边界设为 |x|>0.95, |y|>0.95
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    oob = 0.0
    if abs(cart_x) > 0.95:
        oob += (abs(cart_x) - 0.95) ** 2
    if abs(cart_y) > 0.95:
        oob += (abs(cart_y) - 0.95) ** 2
    components["out_of_bounds"] = -20.0 * oob

    # 5) 首次进入泊位（一次性事件）
    enter_event = 0.0
    if inside > 0.5 and align_gate > 0.5 and slow > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 30.0
    components["first_dock_entry"] = enter_event

    # 6) 完成事件（一次性）
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["dock_success"] = success_event

    # ---------- 完成状态下：除一次性事件外其余组件必须为 0 ----------
    docked_state = (inside > 0.5 and align_gate > 0.5 and slow > 0.5)
    if docked_state:
        components["crate_progress"] = 0.0
        components["soft_contact"] = 0.0
        components["dock_speed_penalty"] = 0.0
        components["out_of_bounds"] = 0.0
        # first_dock_entry 与 dock_success 保留（一次性事件）

    total = 0.0
    for k in components:
        total += components[k]

    return (float(total), components)