# ============================================================================
# 自检记录（估算：dt≈0.05 s / 步；稳定推箱时货箱约 0.4 m/s → 约 0.02 m/步）
#   R_idle    ≈ -0.002                                  （只有 time_cost）
#   R_push    ≈ +0.020 - 0.0005 - 0.002  ≈ +0.0175      （progress +0.02，
#               approach≈0，稳态推箱 closing≈0 → roughness≈0）
#   R_settled ≈ +2.000 - 0.002           ≈ +1.998      （完成谓词成立）
#   自检①：R_push(+0.0175) > R_idle(-0.002) ✓（正常推箱时单步罚项量级≈0.0025）
#   自检②：泊位外 0.3 m 悬停 400 步 ≈ -0.8；真正入坞停稳
#          （dock_enter +5，settled_hold 每步 +2，terminal_success +300）远高于它 ✓
#   自检③：closing=1.0 → roughness -0.150；closing=0.05 → -0.0075；
#          差距 0.1425 ≫ 正常推进项单步 ~0.02，与推进项同量级且更大 ✓
#          （closing≈0 的匀速推箱不被罚 → 不会因为怕罚而不敢推）
#   自检④：同一停稳态连续调用 12 次，每次增量恒为 settled_hold=+2.0 ✓
#          （dock_enter 只发一次；terminal_success 只发一次；均不关闭每步收益）
#   自检⑤：|x|=1.05 → boundary_guard ≈ -1.61 ≪ 0（场地中央=0） ✓
#   量级：1.55 m/s 硬撞时 roughness+hard_hit ≈ -0.73，远大于单步推进收益，
#         高速撞击明确不划算；而稳态推箱几乎不受罚，推进梯度保留。
# ============================================================================

_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_HITS = [0]
_FAILED = [False]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 0) 回合边界检测：obs[18] 单步内单调递增，重置时回落 ----------
    t_now = float(next_obs[18])
    if t_now < _PREV_T[0] or t_now <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _HITS[0] = 0
        _FAILED[0] = False
    _PREV_T[0] = t_now

    # ---------- 1) 几何还原 ----------
    cos_h = obs[2]
    sin_h = obs[3]
    cos_n = next_obs[2]
    sin_n = next_obs[3]

    rel_x = obs[6] * 3.0
    rel_y = obs[7] * 3.0
    rel_nx = next_obs[6] * 3.0
    rel_ny = next_obs[7] * 3.0

    d_cart_crate = (rel_x * rel_x + rel_y * rel_y) ** 0.5
    d_cart_crate_n = (rel_nx * rel_nx + rel_ny * rel_ny) ** 0.5

    dx_dock = obs[12] * 5.0
    dy_dock = obs[13] * 4.0
    dx_dock_n = next_obs[12] * 5.0
    dy_dock_n = next_obs[13] * 4.0
    d_dock = (dx_dock * dx_dock + dy_dock * dy_dock) ** 0.5
    d_dock_n = (dx_dock_n * dx_dock_n + dy_dock_n * dy_dock_n) ** 0.5

    # ---------- 2) 主推进项：有符号、对称（米） ----------
    approach_cargo = 1.0 * (d_cart_crate - d_cart_crate_n)   # 小车→货箱 本帧缩短量
    progress = 1.0 * (d_dock - d_dock_n)                     # 货箱→泊位 本帧缩短量

    # ---------- 3) 接触轻柔度代理（观测里没有冲量，用“接近速度 × 接触”） ----------
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    crate_along = crate_vx * cos_h + crate_vy * sin_h        # 货箱速度沿车头分量
    closing = obs[4] * 3.0 - crate_along                     # 正在接近的速度
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    # 接触中且正在接近才罚；closing≈0 的匀速推箱不罚（否则策略不敢推）
    roughness = -0.15 * contact * closing

    # 硬冲击代理：接触 + 接近速度超过 1.0 m/s
    hard_hit = 0.0
    if contact > 0.5 and closing > 1.0:
        hard_hit = -0.5
        _HITS[0] = _HITS[0] + 1

    # ---------- 4) 动作代价 / 时间代价 ----------
    action_cost = -0.0005 * (action[0] * action[0] + action[1] * action[1])
    time_cost = -0.002

    # ---------- 5) 越界守卫（随越界接近单调下降；+1.0 就是墙） ----------
    over_cart = max(abs(next_obs[0]), abs(next_obs[1])) - 0.90
    if over_cart < 0.0:
        over_cart = 0.0

    crate_wx = next_obs[0] * 5.0 + rel_nx * cos_n - rel_ny * sin_n
    crate_wy = next_obs[1] * 4.0 + rel_nx * sin_n + rel_ny * cos_n
    over_crate = max(abs(crate_wx / 5.0), abs(crate_wy / 4.0)) - 0.90
    if over_crate < 0.0:
        over_crate = 0.0

    boundary_guard = (-10.0 * over_cart - 5.0 * over_cart * over_cart
                      - 10.0 * over_crate - 5.0 * over_crate * over_crate)

    # ---------- 6) 完成侧信号 ----------
    # 容差严格取自环境事实：|obs[12]|<=0.024 且 |obs[13]|<=0.030；朝向误差<30°；速度<0.05 m/s
    inside = (abs(next_obs[12]) <= 0.024) and (abs(next_obs[13]) <= 0.030)
    aligned = next_obs[10] >= 0.8660254
    slow = crate_speed < 0.05
    settled = inside and aligned and slow

    # 首次完全进入泊位容差：整局只发一次
    dock_enter = 0.0
    if inside and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0

    # 完成条件连续计数（不要用“每步给分”当完成事件）
    if settled:
        _STREAK[0] = _STREAK[0] + 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # 停稳期每步收益：只要谓词成立就每步发放，不被一次性事件关闭
    settled_hold = 2.0 if settled else 0.0

    # 失败事件：越界或累计硬冲击 >= 3，整局只发一次
    failure_event = 0.0
    if not _FAILED[0]:
        oob = ((abs(next_obs[0]) > 1.05) or (abs(next_obs[1]) > 1.05)
               or (over_crate > 0.15) or (_HITS[0] >= 3))
        if oob:
            _FAILED[0] = True
            failure_event = -100.0

    # ---------- 7) 汇总 ----------
    total = (approach_cargo + progress + dock_enter + roughness + action_cost
             + time_cost + hard_hit + success_event + failure_event
             + boundary_guard + settled_hold)

    components = {
        "approach_cargo": approach_cargo,
        "progress": progress,
        "dock_enter": dock_enter,
        "roughness": roughness,
        "action_cost": action_cost,
        "time_cost": time_cost,
        "hard_hit": hard_hit,
        "terminal_success": success_event,
        "terminal_failure": failure_event,
        "boundary_guard": boundary_guard,
        "settled_hold": settled_hold,
    }
    return float(total), components