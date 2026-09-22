_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_HARD_HITS = [0]
_FAILED = [False]

_K_ROUGH = 0.15
_HARD_HIT_CLOSING = 0.8
_IN_DOCK_HALF_X = 0.20
_IN_DOCK_HALF_Y = 0.20
_ALIGN_COS = 0.866
_SLOW_SPEED = 0.05
_SLOW_SCALE = 0.30
_DOCKED_W = 3.0
_SETTLE_STEP_BONUS = 1.0
_SUCCESS_BONUS = 150.0
_FAILURE_PENALTY = -60.0


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------------- 0. episode 边界与状态重置 ----------------
    t = float(next_obs[18])
    if t < _PREV_T[0] - 1e-9 or _PREV_T[0] < -0.5:
        _STREAK[0] = 0
        _PAID[0] = False
        _HARD_HITS[0] = 0
        _FAILED[0] = False
    _PREV_T[0] = t

    # ---------------- 1. 几何还原（仅用声明的索引/量纲） ----------------
    cart_x = obs[0] * 5.0
    cart_y = obs[1] * 4.0
    relx = obs[6] * 3.0
    rely = obs[7] * 3.0
    ch = obs[2]
    sh = obs[3]
    crate_x = cart_x + relx * ch - rely * sh
    crate_y = cart_y + relx * sh + rely * ch

    cart_x_n = next_obs[0] * 5.0
    cart_y_n = next_obs[1] * 4.0
    relx_n = next_obs[6] * 3.0
    rely_n = next_obs[7] * 3.0
    ch_n = next_obs[2]
    sh_n = next_obs[3]
    crate_x_n = cart_x_n + relx_n * ch_n - rely_n * sh_n
    crate_y_n = cart_y_n + relx_n * sh_n + rely_n * ch_n

    # ---------------- 2. 有符号的稠密进展（对称，禁止 max(0,.) 刷分） ----------------
    d_cc_prev = ((crate_x - cart_x) ** 2 + (crate_y - cart_y) ** 2) ** 0.5
    d_cc_next = ((crate_x_n - cart_x_n) ** 2 + (crate_y_n - cart_y_n) ** 2) ** 0.5
    approach_cargo = (d_cc_prev - d_cc_next) * 0.5

    dk_x_prev = obs[12] * 5.0
    dk_y_prev = obs[13] * 4.0
    dk_x_next = next_obs[12] * 5.0
    dk_y_next = next_obs[13] * 4.0
    d_dock_prev = (dk_x_prev ** 2 + dk_y_prev ** 2) ** 0.5
    d_dock_next = (dk_x_next ** 2 + dk_y_next ** 2) ** 0.5
    progress = (d_dock_prev - d_dock_next) * 1.5

    # ---------------- 3. 接触柔性代理（仅接触且正在接近时生效） ----------------
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0

    roughness = -_K_ROUGH * contact * closing

    hard_hit = 0.0
    if contact > 0.5 and closing > _HARD_HIT_CLOSING:
        hard_hit = -0.5
        _HARD_HITS[0] += 1

    action_cost = -0.0005 * (action[0] * action[0] + action[1] * action[1])
    time_cost = -0.002

    # ---------------- 4. 完成侧：门控的连续停靠质量（替代瞬态 dock_enter） ----------------
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    in_dock = (abs(dk_x_next) < _IN_DOCK_HALF_X) and (abs(dk_y_next) < _IN_DOCK_HALF_Y)
    aligned_soft = abs(next_obs[10])
    slow_soft = 1.0 - crate_speed / _SLOW_SCALE
    if slow_soft < 0.0:
        slow_soft = 0.0

    docked_bonus = 0.0
    if in_dock:
        docked_bonus = _DOCKED_W * aligned_soft * slow_soft

    aligned = abs(next_obs[10]) >= _ALIGN_COS
    slow = crate_speed < _SLOW_SPEED
    settled = in_dock and aligned and slow

    if settled:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        terminal_success = _SUCCESS_BONUS

    settle_bonus = _SETTLE_STEP_BONUS if settled else 0.0

    # ---------------- 5. 越界守卫 ----------------
    cart_edge = abs(next_obs[0])
    if abs(next_obs[1]) > cart_edge:
        cart_edge = abs(next_obs[1])
    crate_edge = abs(crate_x_n) / 5.0
    if abs(crate_y_n) / 4.0 > crate_edge:
        crate_edge = abs(crate_y_n) / 4.0

    bounds_penalty = 0.0
    if cart_edge > 0.90:
        bounds_penalty = bounds_penalty - 4.0 * (cart_edge - 0.90)
    if crate_edge > 0.90:
        bounds_penalty = bounds_penalty - 4.0 * (crate_edge - 0.90)
    if cart_edge > 1.05:
        bounds_penalty = bounds_penalty - 10.0 * (cart_edge - 1.05)
    if crate_edge > 1.05:
        bounds_penalty = bounds_penalty - 10.0 * (crate_edge - 1.05)

    cart_oob = (abs(next_obs[0]) > 1.05) or (abs(next_obs[1]) > 1.05)
    crate_oob = (abs(crate_x_n) > 5.25) or (abs(crate_y_n) > 4.2)
    terminal_failure = 0.0
    if (cart_oob or crate_oob or _HARD_HITS[0] >= 3) and not _FAILED[0]:
        _FAILED[0] = True
        terminal_failure = _FAILURE_PENALTY

    # ---------------- 6. 汇总 ----------------
    components = {
        "approach_cargo": approach_cargo,
        "progress": progress,
        "docked_bonus": docked_bonus,
        "roughness": roughness,
        "action_cost": action_cost,
        "time_cost": time_cost,
        "hard_hit": hard_hit,
        "settle_bonus": settle_bonus,
        "terminal_success": terminal_success,
        "bounds_penalty": bounds_penalty,
        "terminal_failure": terminal_failure,
    }
    total = 0.0
    for k in components:
        total = total + components[k]
    return float(total), components