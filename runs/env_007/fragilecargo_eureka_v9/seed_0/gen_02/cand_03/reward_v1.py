def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------------- 0. 回合边界检测（obs[18] 单调递增，重置时回落） ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 0.0025:
        _STREAK[0] = 0
        _PAID_SUCCESS[0] = False
        _ENTERED[0] = False
        _HARD_HITS[0] = 0
        _PAID_FAIL[0] = False
    _PREV_T[0] = t

    # ---------------- 1. 几何还原（单位：米） ----------------
    cart_x = obs[0] * 5.0
    cart_y = obs[1] * 4.0
    c = obs[2]
    s = obs[3]

    rel_x = obs[6] * 3.0
    rel_y = obs[7] * 3.0
    d_cart_crate = (rel_x * rel_x + rel_y * rel_y) ** 0.5
    nrel_x = next_obs[6] * 3.0
    nrel_y = next_obs[7] * 3.0
    nd_cart_crate = (nrel_x * nrel_x + nrel_y * nrel_y) ** 0.5

    ddx = obs[12] * 5.0
    ddy = obs[13] * 4.0
    d_crate_dock = (ddx * ddx + ddy * ddy) ** 0.5
    nddx = next_obs[12] * 5.0
    nddy = next_obs[13] * 4.0
    nd_crate_dock = (nddx * nddx + nddy * nddy) ** 0.5

    crate_x = cart_x + c * rel_x - s * rel_y
    crate_y = cart_y + s * rel_x + c * rel_y

    # ---------------- 2. 接触 / 速度量 ----------------
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    crate_along = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0

    # ---------------- 3. 主信号：货箱 -> 泊位的净进展（被高速逼近软门控） -------
    progress_raw = 4.0 * (d_crate_dock - nd_crate_dock)
    ram = closing - 0.30
    if ram < 0.0:
        ram = 0.0
    gate = 1.0 / (1.0 + 1.5 * ram)
    if contact > 0.5:
        progress = progress_raw * gate
        roughness = progress_raw * (gate - 1.0)
    else:
        progress = progress_raw
        roughness = 0.0

    # ---------------- 4. 接近货箱的引导（势函数差分，有界、不可刷） -------------
    approach_cargo = 0.8 * (d_cart_crate - nd_cart_crate)

    # ---------------- 5. 停靠势函数 Φ 的差分塑形（状态不变时恒为 0） -----------
    near_f = 1.0 - nd_crate_dock / 0.70
    if near_f < 0.0:
        near_f = 0.0
    slow_f = 1.0 - crate_speed / 0.50
    if slow_f < 0.0:
        slow_f = 0.0
    na = next_obs[10]
    if na < 0.0:
        na = -na
    align_f = (na - 0.65) / 0.35
    if align_f < 0.0:
        align_f = 0.0
    if align_f > 1.0:
        align_f = 1.0
    phi_next = (near_f * slow_f * align_f) ** (1.0 / 3.0)

    p_near = 1.0 - d_crate_dock / 0.70
    if p_near < 0.0:
        p_near = 0.0
    prev_vx = obs[8] * 3.0
    prev_vy = obs[9] * 3.0
    prev_speed = (prev_vx * prev_vx + prev_vy * prev_vy) ** 0.5
    p_slow = 1.0 - prev_speed / 0.50
    if p_slow < 0.0:
        p_slow = 0.0
    pa = obs[10]
    if pa < 0.0:
        pa = -pa
    p_align = (pa - 0.65) / 0.35
    if p_align < 0.0:
        p_align = 0.0
    if p_align > 1.0:
        p_align = 1.0
    phi_prev = (p_near * p_slow * p_align) ** (1.0 / 3.0)

    settle_shaping = 1.5 * (phi_next - phi_prev)

    # ---------------- 6. 原地不动抑制（车与箱都静止且远离泊位时才生效） --------
    cart_speed_abs = obs[4] * 3.0
    if cart_speed_abs < 0.0:
        cart_speed_abs = -cart_speed_abs
    stall = 0.0
    if crate_speed < 0.06 and cart_speed_abs < 0.15 and nd_crate_dock > 1.0:
        stall = -0.004

    # ---------------- 7. 轻量成本 ----------------
    action_cost = -0.0004 * (action[0] * action[0] + action[1] * action[1])
    time_cost = -0.003

    # ---------------- 8. 首次进入泊位邻域：一次性 +5 ----------------
    dock_enter = 0.0
    if nd_crate_dock < 0.45 and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0

    # ---------------- 9. 硬冲击（接触 + 极高接近速度） ----------------
    hard_hit = 0.0
    if contact > 0.5 and closing > 1.5:
        hard_hit = -0.5
        _HARD_HITS[0] += 1

    # ---------------- 10. 停稳完成谓词（比环境判据更严） ----------------
    tight = 0.0
    if abs(nddx) < 0.14 and abs(nddy) < 0.11 and na >= 0.92 and crate_speed < 0.035:
        tight = 1.0
    if tight > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0
    terminal_success = 0.0
    if _STREAK[0] >= 12 and not _PAID_SUCCESS[0]:
        _PAID_SUCCESS[0] = True
        terminal_success = 200.0

    # ---------------- 11. 越界守卫（车按归一化，箱按世界坐标 / 半宽半高） -------
    cart_margin = abs(obs[0])
    if abs(obs[1]) > cart_margin:
        cart_margin = abs(obs[1])
    crate_margin = abs(crate_x) / 5.0
    if abs(crate_y) / 4.0 > crate_margin:
        crate_margin = abs(crate_y) / 4.0

    m = cart_margin
    if crate_margin > m:
        m = crate_margin
    bounds_safety = 0.0
    if m > 0.90:
        over = m - 0.90
        bounds_safety = -(2.0 * over + 100.0 * over * over)

    terminal_failure = 0.0
    if not _PAID_FAIL[0]:
        if cart_margin > 1.05 or crate_margin > 1.0 or _HARD_HITS[0] >= 3:
            _PAID_FAIL[0] = True
            terminal_failure = -100.0

    # ---------------- 12. 汇总 ----------------
    components = {
        "progress": progress,
        "roughness": roughness,
        "approach_cargo": approach_cargo,
        "settle_shaping": settle_shaping,
        "dock_enter": dock_enter,
        "hard_hit": hard_hit,
        "stall_penalty": stall,
        "action_cost": action_cost,
        "time_cost": time_cost,
        "bounds_safety": bounds_safety,
        "terminal_success": terminal_success,
        "terminal_failure": terminal_failure,
    }

    total = (progress + roughness + approach_cargo + settle_shaping + dock_enter
             + hard_hit + stall + action_cost + time_cost + bounds_safety
             + terminal_success + terminal_failure)

    return (float(total), components)