_PREV_T = [-1.0]     # 上一帧 obs[18]，用于检测 episode 边界
_STREAK = [0]        # "泊位内 + 对齐 + 慢" 连续成立步数
_PAID = [False]      # 成功奖励是否已发放
_FAILED = [False]    # 失败惩罚是否已发放
_HARD_HITS = [0]     # 硬冲击代理计数

_W_APPROACH = 0.4        # cart->crate 势函数增量权重
_W_PROGRESS = 1.5        # crate->dock 势函数增量权重（主运输信号）
_SHAPE_W = 1.5           # 停靠势函数幅度（差值形式，总量有界，不可刷分）
_SHAPE_BOX = 0.50        # m，停靠势的"近泊位"尺度
_SHAPE_V = 0.60          # m/s，停靠势的"慢"尺度
_GENTLE_CLOSING = 0.35   # m/s，超过此接近速度才开始罚（hinge）
_K_ROUGH = 0.6           # hinge 斜率
_HARD_HIT_CLOSING = 0.8  # m/s，硬冲击代理阈值
_IN_DOCK_HALF = 0.25     # m，判定"进泊位"的方盒半宽（保守，不宽于真实泊位）
_ALIGN_COS = 0.866       # cos(30 deg)
_SLOW_SPEED = 0.05       # m/s
_HOLD_BONUS = 0.5        # 每步停稳收益（环境在连续 10 步后立即终止）
_STREAK_NEED = 8         # 略早于环境的 10 步，保证终点奖励能被发放
_SUCCESS_BONUS = 100.0
_FAILURE_PENALTY = -60.0


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 0. 回合边界 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _FAILED[0] = False
        _HARD_HITS[0] = 0
    _PREV_T[0] = t

    # ---------- 1. 几何还原（只用环境声明索引/量纲）----------
    ch = obs[2]
    sh = obs[3]
    relx = obs[6] * 3.0
    rely = obs[7] * 3.0
    cart_x = obs[0] * 5.0
    cart_y = obs[1] * 4.0
    crate_x = cart_x + relx * ch - rely * sh
    crate_y = cart_y + relx * sh + rely * ch

    ch_n = next_obs[2]
    sh_n = next_obs[3]
    relx_n = next_obs[6] * 3.0
    rely_n = next_obs[7] * 3.0
    cart_x_n = next_obs[0] * 5.0
    cart_y_n = next_obs[1] * 4.0
    crate_x_n = cart_x_n + relx_n * ch_n - rely_n * sh_n
    crate_y_n = cart_y_n + relx_n * sh_n + rely_n * ch_n

    # ---------- 2. 运输势函数（有符号差值，不可原地刷分）----------
    d_cc_prev = ((crate_x - cart_x) ** 2 + (crate_y - cart_y) ** 2) ** 0.5
    d_cc_next = ((crate_x_n - cart_x_n) ** 2 + (crate_y_n - cart_y_n) ** 2) ** 0.5
    approach_cargo = _W_APPROACH * (d_cc_prev - d_cc_next)

    dx_prev = obs[12] * 5.0
    dy_prev = obs[13] * 4.0
    dx_next = next_obs[12] * 5.0
    dy_next = next_obs[13] * 4.0
    d_dock_prev = (dx_prev ** 2 + dy_prev ** 2) ** 0.5
    d_dock_next = (dx_next ** 2 + dy_next ** 2) ** 0.5
    progress = _W_PROGRESS * (d_dock_prev - d_dock_next)

    # ---------- 3. 停靠势函数（近泊位 x 慢 x 对齐，差值形式，无全局持续奖励）----------
    sp_prev = ((obs[8] * 3.0) ** 2 + (obs[9] * 3.0) ** 2) ** 0.5
    sp_next = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5

    pe = abs(dx_prev)
    if abs(dy_prev) > pe:
        pe = abs(dy_prev)
    box_prev = 1.0 - pe / _SHAPE_BOX
    if box_prev < 0.0:
        box_prev = 0.0
    slow_prev = 1.0 - sp_prev / _SHAPE_V
    if slow_prev < 0.0:
        slow_prev = 0.0
    al_prev = (abs(obs[10]) - 0.40) / 0.60
    if al_prev < 0.0:
        al_prev = 0.0
    phi_prev = _SHAPE_W * box_prev * slow_prev * al_prev

    ne = abs(dx_next)
    if abs(dy_next) > ne:
        ne = abs(dy_next)
    box_next = 1.0 - ne / _SHAPE_BOX
    if box_next < 0.0:
        box_next = 0.0
    slow_next = 1.0 - sp_next / _SHAPE_V
    if slow_next < 0.0:
        slow_next = 0.0
    al_next = (abs(next_obs[10]) - 0.40) / 0.60
    if al_next < 0.0:
        al_next = 0.0
    phi_next = _SHAPE_W * box_next * slow_next * al_next

    settle_shape = phi_next - phi_prev

    # ---------- 4. 接触安全（hinge，仅在高速接近时生效）----------
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    crate_along = next_obs[8] * 3.0 * obs[2] + next_obs[9] * 3.0 * obs[3]
    closing = obs[4] * 3.0 - crate_along
    if closing < 0.0:
        closing = 0.0

    roughness = 0.0
    if contact > 0.5 and closing > _GENTLE_CLOSING:
        roughness = -_K_ROUGH * (closing - _GENTLE_CLOSING)

    hard_hit = 0.0
    if contact > 0.5 and closing > _HARD_HIT_CLOSING:
        hard_hit = -0.4
        _HARD_HITS[0] += 1

    # ---------- 5. 停稳判定 + 完成奖励 ----------
    in_dock = (abs(dx_next) < _IN_DOCK_HALF) and (abs(dy_next) < _IN_DOCK_HALF)
    aligned = abs(next_obs[10]) >= _ALIGN_COS
    slow = sp_next < _SLOW_SPEED
    settled = in_dock and aligned and slow

    if settled:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    hold_bonus = _HOLD_BONUS if settled else 0.0

    success_bonus = 0.0
    if _STREAK[0] >= _STREAK_NEED and not _PAID[0]:
        _PAID[0] = True
        success_bonus = _SUCCESS_BONUS

    # ---------- 6. 常规小罚项 ----------
    action_cost = -0.0005 * (action[0] * action[0] + action[1] * action[1])
    time_cost = -0.002

    # ---------- 7. 越界守卫 ----------
    cart_edge = abs(next_obs[0])
    if abs(next_obs[1]) > cart_edge:
        cart_edge = abs(next_obs[1])
    crate_edge = abs(crate_x_n) / 5.0
    if abs(crate_y_n) / 4.0 > crate_edge:
        crate_edge = abs(crate_y_n) / 4.0

    bounds_penalty = 0.0
    if cart_edge > 0.92:
        bounds_penalty = bounds_penalty - 3.0 * (cart_edge - 0.92)
    if crate_edge > 0.92:
        bounds_penalty = bounds_penalty - 3.0 * (crate_edge - 0.92)
    if cart_edge > 1.05:
        bounds_penalty = bounds_penalty - 8.0 * (cart_edge - 1.05)
    if crate_edge > 1.05:
        bounds_penalty = bounds_penalty - 8.0 * (crate_edge - 1.05)
    if bounds_penalty < -30.0:
        bounds_penalty = -30.0

    cart_oob = (abs(next_obs[0]) > 1.05) or (abs(next_obs[1]) > 1.05)
    crate_oob = (abs(crate_x_n) > 5.25) or (abs(crate_y_n) > 4.2)
    failure_penalty = 0.0
    if (cart_oob or crate_oob or _HARD_HITS[0] >= 3) and not _FAILED[0]:
        _FAILED[0] = True
        failure_penalty = _FAILURE_PENALTY

    # ---------- 8. 汇总 ----------
    components = {
        "approach_cargo": approach_cargo,
        "progress": progress,
        "settle_shape": settle_shape,
        "hold_bonus": hold_bonus,
        "roughness": roughness,
        "hard_hit": hard_hit,
        "action_cost": action_cost,
        "time_cost": time_cost,
        "bounds_penalty": bounds_penalty,
        "failure_penalty": failure_penalty,
        "success_bonus": success_bonus,
    }
    total = 0.0
    for k in components:
        total = total + components[k]
    return float(total), components