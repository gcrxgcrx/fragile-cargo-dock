```python
# 模块级状态（必须在函数之前声明）
_PREV_T = [-1.0]      # 上一步的 time_fraction，用于检测 episode 边界
_STREAK = [0]         # 完成谓词连续成立计数
_PAID = [False]       # terminal_success 是否已发放
_ENTERED = [False]    # 货箱是否曾经进入泊位容差
_FAILED = [False]     # terminal_failure 是否已发放
_HITS = [0]           # 硬冲击次数（代理计数）
_IN_HIT = [False]     # 上一步是否处于硬冲击中（用于上升沿计数）

# ---------------------------------------------------------------------------
# 自检记录（用本函数自身估算，dt 约 0.05 s）
#   ① idle（静止、无接触）        : time_cost           ≈ -0.002
#      push（接触、匀速推箱推进）  : progress(+0.025) + roughness(-0.010)
#                                     + time(-0.002) + action(-0.0005) ≈ +0.012
#      => push > idle  ✔（差距 0.014，大于正常推进时活跃罚项 0.010）
#   ② 悬停泊位外 0.3 m 保持 400 步 : 每步 -0.002  => 累计 ≈ -0.8
#      真正入坞停稳 10 步          : dock_settle 10×1.0 + dock_enter +5
#                                     + terminal_success +300 => ≈ +315  ✔
#   ③ ⑤ closing=1.0 vs ⑥ closing=0.05（均接触）:
#      差 ≈ 0.2×0.95 (roughness) + 0.5 (hard_hit) ≈ 0.69
#      ≫ 推进项正常单步值(~0.05)  ✔
#   ④ 同一停稳状态连续调用 12 次：每步差值恒为 +1.0（第 10 步另加 +300）✔
#   ⑤ R_idle ≈ -0.002 < R_push ≈ +0.012 < R_settled ≈ +1.0  ✔
#      （R_push - R_idle 大于正常推箱时 roughness/action 的单步量级）
# ---------------------------------------------------------------------------


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------------- 1. episode 边界检测 ----------------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _FAILED[0] = False
        _HITS[0] = 0
        _IN_HIT[0] = False
    _PREV_T[0] = t

    # ---------------- 2. 几何量 ----------------
    # 小车 -> 货箱距离（车体坐标系下的相对位移，原始量纲除以 3.0）
    rx_prev = obs[6] * 3.0
    ry_prev = obs[7] * 3.0
    d_cc_prev = (rx_prev * rx_prev + ry_prev * ry_prev) ** 0.5

    rx_next = next_obs[6] * 3.0
    ry_next = next_obs[7] * 3.0
    d_cc_next = (rx_next * rx_next + ry_next * ry_next) ** 0.5

    # 货箱 -> 泊位中心距离（obs[12] 按半宽 5.0 归一化，obs[13] 按半高 4.0）
    ddx_prev = obs[12] * 5.0
    ddy_prev = obs[13] * 4.0
    d_dock_prev = (ddx_prev * ddx_prev + ddy_prev * ddy_prev) ** 0.5

    ddx_next = next_obs[12] * 5.0
    ddy_next = next_obs[13] * 4.0
    d_dock_next = (ddx_next * ddx_next + ddy_next * ddy_next) ** 0.5

    # 势函数形式、有符号、对称（接近为正，远离为负），单位：米
    approach_cargo = d_cc_prev - d_cc_next
    progress = d_dock_prev - d_dock_next

    # ---------------- 3. 轻柔度 / 粗糙度（观测代理） ----------------
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along = crate_vx * obs[2] + crate_vy * obs[3]      # 货箱速度沿车头方向分量
    closing = obs[4] * 3.0 - crate_along                     # 正在接近的速度
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0

    # 只在"接触且正在接近"时生效；系数足以让 1.0 m/s 级别的冲击明显不划算，
    # 但正常匀速推箱时 closing≈0，几乎不产生惩罚（不会压制推进）。
    roughness = -0.2 * contact * closing

    # ---------------- 4. 硬冲击（单步固定惩罚 + 累计计数） ----------------
    hard_now = (contact > 0.5) and (closing >= 1.0)
    if hard_now:
        if not _IN_HIT[0]:
            _HITS[0] += 1
        _IN_HIT[0] = True
    else:
        _IN_HIT[0] = False
    hard_hit = -0.5 if hard_now else 0.0

    # ---------------- 5. 泊位判定（容差严格取自"0.30 m 不算交付"的约束） ----
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    in_dock = 1.0 if d_dock_next < 0.10 else 0.0                 # 完全进入容差
    aligned = 1.0 if next_obs[10] > 0.866 else 0.0               # 朝向误差 < 30°
    slow = 1.0 if crate_speed < 0.05 else 0.0                    # 速度 < 0.05 m/s
    settled = in_dock * aligned * slow

    # 连续计数（用于一次性完成事件）
    if settled > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # 首次完全进入泊位：整局一次性
    dock_enter = 0.0
    if in_dock > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0

    # 连续 10 步满足完成谓词：一次性（环境此时立即终止，故最多发一次）
    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        terminal_success = 300.0

    # 停稳期每步收益：谓词成立就发，不受任何"已发放"开关影响
    dock_settle = 1.0 * settled

    # ---------------- 6. 越界守卫（小车 + 货箱，单调递增的接近惩罚） ----
    cart_ax = abs(obs[0])
    if abs(next_obs[0]) > cart_ax:
        cart_ax = abs(next_obs[0])
    cart_ay = abs(obs[1])
    if abs(next_obs[1]) > cart_ay:
        cart_ay = abs(next_obs[1])

    # 货箱世界坐标 = 小车位置 + 旋转到世界系的相对位移
    bx = obs[6] * 3.0
    by = obs[7] * 3.0
    crate_x = obs[0] * 5.0 + bx * obs[2] - by * obs[3]
    crate_y = obs[1] * 4.0 + bx * obs[3] + by * obs[2]

    bx2 = next_obs[6] * 3.0
    by2 = next_obs[7] * 3.0
    crate_x2 = next_obs[0] * 5.0 + bx2 * next_obs[2] - by2 * next_obs[3]
    crate_y2 = next_obs[1] * 4.0 + bx2 * next_obs[3] + by2 * next_obs[2]

    crate_nx = abs(crate_x) / 5.0
    if abs(crate_x2) / 5.0 > crate_nx:
        crate_nx = abs(crate_x2) / 5.0
    crate_ny = abs(crate_y) / 4.0
    if abs(crate_y2) / 4.0 > crate_ny:
        crate_ny = abs(crate_y2) / 4.0

    cart_over = 0.0
    if cart_ax > 0.90:
        cart_over += cart_ax - 0.90
    if cart_ay > 0.90:
        cart_over += cart_ay - 0.90

    crate_over = 0.0
    if crate_nx > 0.90:
        crate_over += crate_nx - 0.90
    if crate_ny > 0.90:
        crate_over += crate_ny - 0.90

    bounds_guard = -15.0 * (cart_over + crate_over)

    # ---------------- 7. 终止失败（出界 或 累计硬冲击 ≥ 3） ----------------
    out_of_bounds = (cart_ax > 1.05) or (cart_ay > 1.05) or (crate_nx > 1.0) or (crate_ny > 1.0)
    hits_exceeded = _HITS[0] >= 3

    terminal_failure = 0.0
    if (out_of_bounds or hits_exceeded) and not _FAILED[0]:
        _FAILED[0] = True
        terminal_failure = -100.0

    # ---------------- 8. 动作代价 / 时间代价 ----------------
    action_cost = -0.0005 * (action[0] * action[0] + action[1] * action[1])
    time_cost = -0.002

    # ---------------- 9. 汇总 ----------------
    components = {
        "approach_cargo": approach_cargo,
        "progress": progress,
        "dock_enter": dock_enter,
        "roughness": roughness,
        "action_cost": action_cost,
        "time_cost": time_cost,
        "hard_hit": hard_hit,
        "dock_settle": dock_settle,
        "terminal_success": terminal_success,
        "terminal_failure": terminal_failure,
        "bounds_guard": bounds_guard,
    }

    total = (approach_cargo + progress + dock_enter + roughness + action_cost +
             time_cost + hard_hit + dock_settle + terminal_success +
             terminal_failure + bounds_guard)

    return float(total), components
```