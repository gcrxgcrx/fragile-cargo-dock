_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_HARD_HITS = [0]
_FAILED = [False]

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 自检⑤估算（每步平均）：
    # R_idle    ≈ -0.002
    # R_push    ≈ +0.05（正常推进：approach_cargo + progress ≈ 0.05，罚项 ≈ -0.002）
    # R_settled ≈ +1.0（停稳每步 +1.0，time_cost -0.002）
    # 满足 R_push > R_idle 且 R_settled > R_push。

    # 回合边界检测：obs[18] 单调递增，重置时回落
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _HARD_HITS[0] = 0
        _FAILED[0] = False
    _PREV_T[0] = t

    # 恢复货箱世界坐标
    cart_x_m = next_obs[0] * 5.0
    cart_y_m = next_obs[1] * 4.0
    cos_h = next_obs[2]
    sin_h = next_obs[3]
    body_x = next_obs[6] * 3.0
    body_y = next_obs[7] * 3.0
    crate_wx = cart_x_m + cos_h * body_x - sin_h * body_y
    crate_wy = cart_y_m + sin_h * body_x + cos_h * body_y

    # 1. approach_cargo：小车→货箱距离本帧缩短量（有符号、对称）
    dist_prev = ((obs[6] * 3.0) ** 2 + (obs[7] * 3.0) ** 2) ** 0.5
    dist_next = ((next_obs[6] * 3.0) ** 2 + (next_obs[7] * 3.0) ** 2) ** 0.5
    approach_cargo = (dist_prev - dist_next) * 1.0

    # 2. progress：货箱→泊位距离本帧缩短量（有符号、对称）
    prog_prev = ((obs[12] * 5.0) ** 2 + (obs[13] * 4.0) ** 2) ** 0.5
    prog_next = ((next_obs[12] * 5.0) ** 2 + (next_obs[13] * 4.0) ** 2) ** 0.5
    progress = (prog_prev - prog_next) * 1.0

    # 3. dock_enter：货箱首次完全进入泊位容差，一次性 +5
    dock_inside = (abs(next_obs[12]) <= 0.024) and (abs(next_obs[13]) <= 0.030)
    dock_enter = 0.0
    if dock_inside and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0

    # 4. roughness：可观测接触接近速度代理惩罚
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    # 代理系数取 -0.2：closing=1.0 时惩罚 -0.2，与推进项同量级（自检③）。
    # 正常匀速推箱时 closing≈0，不会被误判为撞击。
    roughness = -0.2 * contact * closing

    # 5. action_cost
    action_cost = -0.0005 * (action[0] ** 2 + action[1] ** 2)

    # 6. time_cost
    time_cost = -0.002

    # 7. hard_hit：单步硬冲击代理（closing 超过 1.5 m/s 视为硬冲击）
    hard_hit = 0.0
    if contact > 0.5 and closing > 1.5:
        hard_hit = -0.5
        _HARD_HITS[0] += 1

    # 完成条件：泊位内 + 对齐 + 慢
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    aligned = abs(next_obs[10]) > 0.866          # cos(30°)
    slow = crate_speed < 0.05
    done_cond = dock_inside and aligned and slow

    if done_cond:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # 8. terminal_success：连续 10 步满足完成条件，一次性 +300（整局只发一次）
    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        terminal_success = 300.0

    # 9. 停稳期每步收益：完成谓词成立时每步发放，不受 _PAID 影响
    settle_bonus = 0.0
    if done_cond:
        settle_bonus = 1.0

    # 10. terminal_failure：小车/货箱越界，或累计硬冲击 >= 3 次，一次性 -100
    cart_out = (abs(next_obs[0]) > 1.05) or (abs(next_obs[1]) > 1.05)
    crate_out = (abs(crate_wx) > 5.0) or (abs(crate_wy) > 4.0)
    hard_hit_fail = _HARD_HITS[0] >= 3
    terminal_failure = 0.0
    if (cart_out or crate_out or hard_hit_fail) and not _FAILED[0]:
        _FAILED[0] = True
        terminal_failure = -100.0

    # 11. 越界守卫：随越界接近单调下降
    cart_norm = max(abs(next_obs[0]), abs(next_obs[1]))
    crate_norm = max(abs(crate_wx) / 5.0, abs(crate_wy) / 4.0)
    boundary_penalty = 0.0
    if cart_norm > 0.9:
        boundary_penalty -= 20.0 * (cart_norm - 0.9)
    if crate_norm > 0.9:
        boundary_penalty -= 20.0 * (crate_norm - 0.9)

    total_reward = (
        approach_cargo
        + progress
        + dock_enter
        + roughness
        + action_cost
        + time_cost
        + hard_hit
        + terminal_success
        + settle_bonus
        + terminal_failure
        + boundary_penalty
    )

    components = {
        "approach_cargo": approach_cargo,
        "progress": progress,
        "dock_enter": dock_enter,
        "roughness": roughness,
        "action_cost": action_cost,
        "time_cost": time_cost,
        "hard_hit": hard_hit,
        "terminal_success": terminal_success,
        "settle_bonus": settle_bonus,
        "terminal_failure": terminal_failure,
        "boundary_penalty": boundary_penalty,
    }
    return float(total_reward), components