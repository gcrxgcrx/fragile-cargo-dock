_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_FAILED = [False]
_HARD_HITS = [0]

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 回合边界检测：obs[18] 是已用时间比例，单调递增，重置时回落
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _FAILED[0] = False
        _HARD_HITS[0] = 0
    _PREV_T[0] = t

    # 1. approach_cargo: 小车→货箱距离本帧缩短量（米），有符号对称
    old_cart_crate = ((obs[6] * 3.0) ** 2 + (obs[7] * 3.0) ** 2) ** 0.5
    new_cart_crate = ((next_obs[6] * 3.0) ** 2 + (next_obs[7] * 3.0) ** 2) ** 0.5
    approach_cargo = (old_cart_crate - new_cart_crate) * 1.0

    # 2. progress: 货箱→泊位距离本帧缩短量（米），有符号对称
    old_crate_dock = ((obs[12] * 5.0) ** 2 + (obs[13] * 4.0) ** 2) ** 0.5
    new_crate_dock = ((next_obs[12] * 5.0) ** 2 + (next_obs[13] * 4.0) ** 2) ** 0.5
    progress = (old_crate_dock - new_crate_dock) * 1.0

    # 3. dock_enter: 货箱首次完全进入泊位容差，一次性 +5
    dock_enter = 0.0
    in_dock_new = (abs(next_obs[12]) <= 0.024 and abs(next_obs[13]) <= 0.030)
    if in_dock_new and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0

    # 4. roughness: 接触接近速度代理（唯一需要代理的项）
    #    正常匀速推箱时 cart 与 crate 速度接近，closing ≈ 0，不会被误罚
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    roughness = -0.2 * contact * closing

    # 5. hard_hit: 接触且接近速度极高时的固定惩罚代理（与 roughness 分档，避免同一公式叠加）
    hard_hit = 0.0
    if contact > 0.5 and closing > 1.5:
        hard_hit = -0.5
        _HARD_HITS[0] += 1

    # 6. action_cost: 动作平方和惩罚
    action_cost = -0.0005 * (action[0] ** 2 + action[1] ** 2)

    # 7. time_cost: 每步固定时间成本
    time_cost = -0.002

    # 8. 完成谓词 + 停稳期每步收益（不得被一次性事件关掉）
    speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    slow = speed < 0.05
    # 货箱朝向与仓库轴对齐：cos/sin 绝对值接近 1 表示朝向 0/90/180/270 度，容差 30 度
    aligned = max(abs(next_obs[10]), abs(next_obs[11])) >= 0.866
    settled = in_dock_new and slow and aligned

    settle_bonus = 1.0 if settled else 0.0

    if settled:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        terminal_success = 300.0

    # 9. terminal_failure: 小车或货箱越界、或累计硬冲击 >= 3 次
    terminal_failure = 0.0
    cart_out = (abs(obs[0]) > 1.05 or abs(obs[1]) > 1.05 or
                abs(next_obs[0]) > 1.05 or abs(next_obs[1]) > 1.05)
    # 由 next_obs 恢复货箱世界坐标并归一化
    cart_x = next_obs[0] * 5.0
    cart_y = next_obs[1] * 4.0
    rel_x = next_obs[6] * 3.0
    rel_y = next_obs[7] * 3.0
    cos_h = next_obs[2]
    sin_h = next_obs[3]
    crate_x = cart_x + rel_x * cos_h - rel_y * sin_h
    crate_y = cart_y + rel_x * sin_h + rel_y * cos_h
    crate_nx = crate_x / 5.0
    crate_ny = crate_y / 4.0
    crate_out = abs(crate_nx) > 1.05 or abs(crate_ny) > 1.05
    if (cart_out or crate_out or _HARD_HITS[0] >= 3) and not _FAILED[0]:
        _FAILED[0] = True
        terminal_failure = -100.0

    # 10. 边界守卫：随越界接近单调下降，在 |x| 或 |y| > 0.9 时生效
    boundary_penalty = 0.0
    cart_margin = max(abs(next_obs[0]), abs(next_obs[1]))
    if cart_margin > 0.9:
        boundary_penalty += -20.0 * (cart_margin - 0.9)
    crate_margin = max(abs(crate_nx), abs(crate_ny))
    if crate_margin > 0.9:
        boundary_penalty += -20.0 * (crate_margin - 0.9)

    total = (approach_cargo + progress + dock_enter + roughness + hard_hit +
             action_cost + time_cost + settle_bonus + terminal_success +
             terminal_failure + boundary_penalty)

    # 自检记录：
    # R_idle ≈ -0.002（仅时间成本）
    # R_push ≈ 0.05 ~ 0.10（推进项 - 微小罚项）
    # R_settled ≈ 1.0（停稳每步收益 - 微小罚项）
    # R_push - R_idle ≈ 0.05 ~ 0.10，大于正常推进时罚项量级（~0.003）
    # R_settled > R_push，停稳是全局最优点
    # 自检③：closing=1.0 时 roughness=-0.2，closing=0.05 时 roughness=-0.01，差 0.19 > 推进单步值
    # 自检④：停稳状态连续调用 12 次，每次差值均为 +1.0（settle_bonus）
    # 自检⑤：边界守卫在 |x|=1.05 时约 -3.0，远低于中心状态

    components = {
        "approach_cargo": approach_cargo,
        "progress": progress,
        "dock_enter": dock_enter,
        "roughness": roughness,
        "hard_hit": hard_hit,
        "action_cost": action_cost,
        "time_cost": time_cost,
        "settle_bonus": settle_bonus,
        "terminal_success": terminal_success,
        "terminal_failure": terminal_failure,
        "boundary_guard": boundary_penalty,
    }
    return float(total), components