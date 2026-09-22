_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_HARD_HITS = [0]
_FAILED_PAID = [False]

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 自检记录（估算，基于典型每步距离变化）：
    # R_idle ≈ -0.002（仅时间成本）
    # R_push ≈ +0.35（推进项约 0.4，动作/时间/轻柔代理约 -0.05）
    # R_settled ≈ +1.00（停稳每步 +1.0，时间/动作约 -0.004）
    # 满足 R_push > R_idle，R_settled > R_push。
    # 自检③：closing=1.0 时 roughness≈-0.50，closing=0.05 时≈-0.025，差距 0.475，
    # 与正常推进项（约 0.4/步）同量级。

    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _HARD_HITS[0] = 0
        _FAILED_PAID[0] = False
    _PREV_T[0] = t

    # 小车→货箱距离（米）
    prev_cart_crate = ((obs[6] * 3.0) ** 2 + (obs[7] * 3.0) ** 2) ** 0.5
    next_cart_crate = ((next_obs[6] * 3.0) ** 2 + (next_obs[7] * 3.0) ** 2) ** 0.5
    approach_cargo = 1.0 * (prev_cart_crate - next_cart_crate)

    # 货箱→泊位距离（米）
    prev_dock = ((obs[12] * 5.0) ** 2 + (obs[13] * 4.0) ** 2) ** 0.5
    next_dock = ((next_obs[12] * 5.0) ** 2 + (next_obs[13] * 4.0) ** 2) ** 0.5
    progress = 1.0 * (prev_dock - next_dock)

    # 首次完全进入泊位容差
    dock_enter = 0.0
    if (abs(next_obs[12]) <= 0.024 and abs(next_obs[13]) <= 0.030 and not _ENTERED[0]):
        _ENTERED[0] = True
        dock_enter = 5.0

    # 接触轻柔度代理（roughness）
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    roughness = -0.5 * contact * closing

    # 硬冲击固定惩罚与计数
    hard_hit = 0.0
    if contact > 0.5 and closing > 1.0:
        hard_hit = -0.5
        _HARD_HITS[0] += 1

    # 动作代价与时间成本
    action_cost = -0.0005 * (action[0] ** 2 + action[1] ** 2)
    time_cost = -0.002

    # 完成谓词：泊位内 + 对齐 + 慢
    speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    align = max(abs(next_obs[10]), abs(next_obs[11]))
    in_dock = (abs(next_obs[12]) <= 0.024 and abs(next_obs[13]) <= 0.030)
    done_now = in_dock and (align >= 0.8660254) and (speed < 0.05)

    # 停稳期每步收益（不因一次性事件或连续计数而关闭）
    settle_bonus = 1.0 if done_now else 0.0

    # 连续 10 步完成计数
    if done_now:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        terminal_success = 300.0

    # 边界守卫（小车位置归一化，超过 0.95 开始惩罚）
    pen_x = max(0.0, abs(next_obs[0]) - 0.95)
    pen_y = max(0.0, abs(next_obs[1]) - 0.95)
    boundary_guard = -50.0 * (pen_x + pen_y)

    # 失败一次性：越界或累计硬冲击 >= 3
    terminal_failure = 0.0
    if not _FAILED_PAID[0]:
        cart_out = (abs(next_obs[0]) > 1.05 or abs(next_obs[1]) > 1.05)

        cart_x = next_obs[0] * 5.0
        cart_y = next_obs[1] * 4.0
        cos_h = next_obs[2]
        sin_h = next_obs[3]
        rel_x = next_obs[6] * 3.0
        rel_y = next_obs[7] * 3.0
        crate_x = cart_x + rel_x * cos_h - rel_y * sin_h
        crate_y = cart_y + rel_x * sin_h + rel_y * cos_h
        crate_out = (abs(crate_x) > 5.25 or abs(crate_y) > 4.2)

        if cart_out or crate_out or _HARD_HITS[0] >= 3:
            _FAILED_PAID[0] = True
            terminal_failure = -100.0

    total = (approach_cargo + progress + dock_enter + roughness + action_cost +
             time_cost + hard_hit + terminal_success + terminal_failure +
             boundary_guard + settle_bonus)

    components = {
        "approach_cargo": approach_cargo,
        "progress": progress,
        "dock_enter": dock_enter,
        "roughness": roughness,
        "action_cost": action_cost,
        "time_cost": time_cost,
        "hard_hit": hard_hit,
        "terminal_success": terminal_success,
        "terminal_failure": terminal_failure,
        "boundary_guard": boundary_guard,
        "settle_bonus": settle_bonus,
    }
    return float(total), components