分析：当前奖励被 `settle_bonus`（≈2.0/步、无上限）和由它触发的 `terminal_success` 完全主导（两项合计 ≈368/局，而真正的运输信号 `progress` 只有 3.15/局），20 个 eval 回合全部 truncation、0 终止，说明这个"停稳"代理判据比环境的成功判据松得多，策略学会了在伪泊位状态里原地收分而不是完成任务。我要做四件事：① 把无上限的每步停稳奖励改成**势函数差分塑形**（状态不变恒为 0，无法靠悬停刷分）；② 把运输进展权重提高一个量级，使其成为主梯度；③ 把 `roughness` 从独立惩罚改成对 progress 的**软门控**（两者之和 ≥ 0，推进永不被净扣分）；④ 收紧完成谓词（按 x/y 轴分解 + 更严朝向/速度 + 更长连续步数），并把一次性完成奖励降到 200，避免假阳性被高额支付。

# ============================================================================
# 自检记录（单步奖励，单位：奖励/步；几何：仓库半宽 5.0 m、半高 4.0 m，
#   obs[6..9] 除以 3.0，obs[12]/obs[13] 分别按 5.0 / 4.0 归一化）
#
#   ① 什么都不做（货箱静止在初始位置，远离泊位，无接触，动作≈0）
#      progress=0, roughness=0, approach=0, settle_shaping=0, dock_enter=0,
#      hard_hit=0, stall=-0.004, action_cost≈0, time_cost=-0.003, bounds=0
#      → R_idle ≈ -0.007
#
#   ② 正在把货箱推向泊位（接触、货箱因此有速度、接近速度≈0）
#      progress = 4.0 * 0.01 ≈ +0.040（本步向泊位缩短 1 cm）
#      roughness = 0（匀速推箱 closing≈0 → gate=1）
#      approach≈0, settle_shaping≈0（离泊位远）, stall=0（货箱在动）
#      action_cost≈-0.0005, time_cost=-0.003
#      → R_push ≈ +0.037
#
#   R_idle(-0.007) < R_push(+0.037)  → 严格成立 ✓（差值约 5e-2，是 idle 罚项的 6 倍）
#
#   门控一致性：contact>0.5 时 progress = raw*gate (0≤gate≤1)、
#   roughness = raw*(gate-1) ≤ 0，两者之和 = raw*gate ≥ 0 →
#   "把货箱推向泊位"在任何情况下都不会被这两个组件净扣分 ✓
#   最大冲击（closing→大）时 gate→0，progress+roughness→0，仅不奖励，不惩罚 ✓
#
#   停稳塑形 Φ=(near*slow*align)**(1/3) ∈ [0,1]，settle_shaping=1.5*(Φ_next-Φ_prev)
#   为势函数差分：同一状态连续重复 → 每步恒为 0，不存在"停在泊位附近持续收分" ✓
#   整局该项被 |Φ|≤1 界住（≤1.5/局），不可能压过 progress 与 terminal_success ✓
#
#   完成谓词（比环境成功条件更严，防止代理误判为成功）：
#     |next_obs[12]*5.0| < 0.14 m 且 |next_obs[13]*4.0| < 0.11 m 且
#     |crate_cos_heading| ≥ 0.92（<23°，环境要求 30°）且 crate 速度 < 0.035 m/s（环境 0.05）
#     连续 12 步（环境 10 步）成立 → 一次性 +200，整局只发一次。
# ============================================================================

_PREV_T = [-1.0]
_STREAK = [0]
_PAID_SUCCESS = [False]
_ENTERED = [False]
_HARD_HITS = [0]
_PAID_FAIL = [False]


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