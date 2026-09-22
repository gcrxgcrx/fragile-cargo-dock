_PREV_T = [-1.0]
_STREAK = [0]
_PAID_SUCCESS = [False]
_ENTERED = [False]
_HARD_HITS = [0]
_PAID_FAIL = [False]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------------- 0. 回合边界检测（obs[18] 单调递增，重置时回落） -------------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 0.0025:
        _STREAK[0] = 0
        _PAID_SUCCESS[0] = False
        _ENTERED[0] = False
        _HARD_HITS[0] = 0
        _PAID_FAIL[0] = False
    _PREV_T[0] = t

    # ---------------- 1. 由 obs 还原几何量（米） ----------------
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

    # ---------------- 2. 分项 ----------------

    # 2.1 approach_cargo：早期引导小车接近货箱（+0.5 / 米）
    approach_cargo = 0.5 * (d_cart_crate - nd_cart_crate)

    # 2.2 progress：主运输信号，货箱->泊位距离缩短量（+2.0 / 米，对称）
    progress = 2.0 * (d_crate_dock - nd_crate_dock)

    # 2.3 dock_enter：首次进入泊位 0.5 m 邻域，整局一次 +5
    dock_enter = 0.0
    if nd_crate_dock < 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0

    # 2.4 结算侧状态量（全部来自 next_obs）
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    align_cos = next_obs[10]
    if align_cos < 0.0:
        align_cos = -align_cos
    contact = 1.0 if next_obs[14] > 0.5 else 0.0

    # 2.5 settle_bonus：联合连续条件（near * slow * align * free），几何平均。
    #     near 随货箱靠近泊位中心平滑上升；slow 随箱速下降；
    #     align 随箱体朝向对齐上升；free 要求车体不再顶着箱子（真正停稳）。
    #     平移/推行阶段 near≈0 或 free≈0，故本项对推动不产生负贡献。
    near_f = 1.0 - nd_crate_dock / 0.6
    if near_f < 0.0:
        near_f = 0.0
    elif near_f > 1.0:
        near_f = 1.0
    slow_f = 1.0 / (1.0 + 8.0 * crate_speed)
    align_f = (align_cos - 0.5) / 0.5
    if align_f < 0.0:
        align_f = 0.0
    elif align_f > 1.0:
        align_f = 1.0
    free_f = 1.0 - contact
    settle_bonus = 0.4 * (near_f * slow_f * align_f * free_f) ** 0.25

    # 2.6 roughness：接触冲量比例代理，只在“接触且正在接近”时为负
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along
    if closing < 0.0:
        closing = 0.0
    roughness = -0.30 * contact * closing

    # 2.7 hard_hit：单步硬冲击（接触 + 高速接近）
    hard_hit = 0.0
    if contact > 0.5 and closing > 1.0:
        hard_hit = -0.5
        _HARD_HITS[0] += 1

    # 2.8 action_cost / time_cost
    action_cost = -0.0005 * (action[0] * action[0] + action[1] * action[1])
    time_cost = -0.002

    # 2.9 terminal_success：紧凑“完全入泊 + 对齐 + 静止 + 不再被顶”谓词，
    #     连续 10 步成立，整局一次 +200。
    ok = 0.0
    if nd_crate_dock < 0.12 and align_cos >= 0.866 and crate_speed < 0.05 and contact < 0.5:
        ok = 1.0
    if ok > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0
    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID_SUCCESS[0]:
        _PAID_SUCCESS[0] = True
        terminal_success = 200.0

    # 2.10 越界度量
    cart_margin = abs(obs[0])
    if abs(obs[1]) > cart_margin:
        cart_margin = abs(obs[1])
    crate_margin = abs(crate_x) / 5.0
    if abs(crate_y) / 4.0 > crate_margin:
        crate_margin = abs(crate_y) / 4.0

    # 2.11 terminal_failure：越界 或 累计硬冲击 >= 3，整局一次 -100
    terminal_failure = 0.0
    if not _PAID_FAIL[0]:
        if cart_margin > 1.05 or crate_margin > 1.0 or _HARD_HITS[0] >= 3:
            _PAID_FAIL[0] = True
            terminal_failure = -100.0

    # 2.12 bounds_safety：随越界接近单调下降的守卫
    m = cart_margin
    if crate_margin > m:
        m = crate_margin
    bounds_safety = 0.0
    if m > 0.90:
        over = m - 0.90
        bounds_safety = -(2.0 * over + 100.0 * over * over)

    # ---------------- 3. 汇总 ----------------
    components = {
        "approach_cargo": approach_cargo,
        "progress": progress,
        "dock_enter": dock_enter,
        "roughness": roughness,
        "action_cost": action_cost,
        "time_cost": time_cost,
        "hard_hit": hard_hit,
        "settle_bonus": settle_bonus,
        "bounds_safety": bounds_safety,
        "terminal_success": terminal_success,
        "terminal_failure": terminal_failure,
    }

    total = (approach_cargo + progress + dock_enter + roughness + action_cost
             + time_cost + hard_hit + settle_bonus + bounds_safety
             + terminal_success + terminal_failure)

    return (float(total), components)