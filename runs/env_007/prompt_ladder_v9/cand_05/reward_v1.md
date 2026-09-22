```python
# ===========================================================================
# 模块级状态（仅用于一次性事件 / 连续计数 / 回合边界检测）
# 无 import、无 class、无 try/except、无 lambda、无 global
# ===========================================================================
_PREV_T = [-1.0]        # 上一帧 time_fraction，用于检测新 episode（回落 or 回到 0）
_STREAK = [0]           # "泊位内 + 对齐 + 慢" 连续步数
_PAID = [False]         # 一次性完成事件是否已发放（整局只发一次）
_ENTERED = [False]      # 是否曾经首次完全进入泊位容差（整局只发一次）
_FAILED = [False]       # 一次性失败事件是否已发放（整局只发一次）
_HIT_COUNT = [0]        # 硬冲击代理计数（用于 >=3 判失败）
_HIT_LATCH = [False]    # 硬冲击去抖：一次连续高速接近只算一次

# ===========================================================================
# 自检记录（量级估算）
#   自检① 什么都不做（货箱静止在初始位置）: R_idle   ≈ -0.003 / 步
#         正常把货箱推向泊位（稳态推箱，closing≈0）: R_push ≈ +0.047 / 步
#         -> R_push - R_idle ≈ 0.05  ≥ 常规单步罚项最大量级(0.003)   通过
#   自检② 泊位外 0.3 m 悬停 400 步: 累计 ≈ -0.8
#         真正入坞并停稳 10 步: 0.25*10 + 5(dock_enter) + 300(实裁到20) ≈ +27.5  通过
#   自检③ closing=1.0 m/s: roughness=-0.20;  closing=0.05: roughness=-0.01
#         差值 0.19  >  1.0 × 推进项单步值(≈0.05)                    通过
#   自检④ 同一"停稳"状态连续调用 12 次: 每次差值恒为 +0.248
#         (= settle_hold 0.25 - time_cost 0.002)，第 10 次额外一次性成功事件  通过
#   自检⑤ R_idle ≈ -0.003 < R_push ≈ +0.047 < R_settled ≈ +0.248
#         （停稳必须是全局最优点；推进项必须显著优于不动）
#   边界守卫: |x|=0.0 -> 0.0 ; |x|=0.95 -> -0.75 ; |x|=1.05 -> -3.6  单调下降
# ===========================================================================


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------------
    # 0) 回合边界检测：time_fraction 单调递增，重置时回落（或回到 0）
    # ------------------------------------------------------------------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _FAILED[0] = False
        _HIT_COUNT[0] = 0
        _HIT_LATCH[0] = False
    _PREV_T[0] = t

    # ------------------------------------------------------------------
    # 1) 几何还原（只用已声明的 obs 维度）
    # ------------------------------------------------------------------
    # 小车 -> 货箱 距离（车体系相对位姿的模长 = 世界系距离，旋转不变）
    rel_x0 = obs[6] * 3.0
    rel_y0 = obs[7] * 3.0
    rel_x1 = next_obs[6] * 3.0
    rel_y1 = next_obs[7] * 3.0
    d_cart_crate_prev = (rel_x0 * rel_x0 + rel_y0 * rel_y0) ** 0.5
    d_cart_crate_next = (rel_x1 * rel_x1 + rel_y1 * rel_y1) ** 0.5

    # 货箱 -> 泊位 距离（obs[12]/obs[13] 是除以半宽/半高的有符号偏移）
    dx0 = obs[12] * 5.0
    dy0 = obs[13] * 4.0
    dx1 = next_obs[12] * 5.0
    dy1 = next_obs[13] * 4.0
    d_dock_prev = (dx0 * dx0 + dy0 * dy0) ** 0.5
    d_dock_next = (dx1 * dx1 + dy1 * dy1) ** 0.5

    # ------------------------------------------------------------------
    # 2) 主推进项（有符号、对称：靠近给正、远离给负，按米计）
    # ------------------------------------------------------------------
    approach_cargo = 1.0 * (d_cart_crate_prev - d_cart_crate_next)
    progress = 1.0 * (d_dock_prev - d_dock_next)

    # ------------------------------------------------------------------
    # 3) 轻柔度：接近速度 × 接触 代理（抑制硬撞击，不压制推进）
    #    稳态推箱时 cart 速度 ≈ 货箱沿车头速度 -> closing ≈ 0，不产生惩罚；
    #    只有"撞上去"（接近速度大）时才显著为负。
    # ------------------------------------------------------------------
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0

    roughness = -0.2 * contact * closing          # 接触 & 接近才罚；k=0.2 使 1.0 m/s
                                                  # 撞击罚 0.20，明显压过推进收益

    # 硬冲击（代理）：接触中接近速度 > 1.0 m/s，去抖后每"一次"只算一次
    hard_hit = 0.0
    if contact > 0.5 and closing > 1.0:
        if not _HIT_LATCH[0]:
            _HIT_LATCH[0] = True
            _HIT_COUNT[0] += 1
            hard_hit = -0.5
    else:
        if closing < 0.4:
            _HIT_LATCH[0] = False

    # ------------------------------------------------------------------
    # 4) 完成侧信号
    #    inside : |obs[12]| <= 0.024 且 |obs[13]| <= 0.030（环境给出的容差）
    #    aligned: 货箱朝向误差 < 30°（cos 阈值，避免 atan2/import）
    #    slow   : 货箱速度 < 0.05 m/s
    # ------------------------------------------------------------------
    inside = 1.0 if (abs(next_obs[12]) <= 0.024 and abs(next_obs[13]) <= 0.030) else 0.0
    aligned = 1.0 if next_obs[10] >= 0.8660254 else 0.0      # cos(30°)
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    slow = 1.0 if crate_speed < 0.05 else 0.0

    # dock_enter：整局只发一次的"首次完全进入容差"
    dock_enter = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0

    # 连续计数（不因一次性事件发放过而关闭）
    if inside > 0.5 and aligned > 0.5 and slow > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # 停稳期每步收益：谓词成立就每步发放，不被任何其它逻辑关掉
    # （环境在该谓词连续 10 步时会立刻结束 episode，因此最多 10 步，无法长期刷分）
    settle_hold = 0.25 if (inside > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0

    # 一次性完成事件：连续 10 步
    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        terminal_success = 300.0

    # ------------------------------------------------------------------
    # 5) 越界守卫（小车 & 货箱）——随越界接近单调下降
    # ------------------------------------------------------------------
    ex_cart = max(abs(obs[0]), abs(obs[1]), abs(next_obs[0]), abs(next_obs[1]))

    c1 = next_obs[2]
    s1 = next_obs[3]
    rx = next_obs[6] * 3.0
    ry = next_obs[7] * 3.0
    crate_wx = next_obs[0] * 5.0 + rx * c1 - ry * s1
    crate_wy = next_obs[1] * 4.0 + rx * s1 + ry * c1
    ex_crate = max(abs(crate_wx) / 5.0, abs(crate_wy) / 4.0)

    bounds_guard = 0.0
    bounds_guard -= 15.0 * max(0.0, ex_cart - 0.90)
    bounds_guard -= 45.0 * max(0.0, ex_cart - 1.02)
    bounds_guard -= 12.0 * max(0.0, ex_crate - 0.90)
    bounds_guard -= 36.0 * max(0.0, ex_crate - 1.02)

    # 一次性失败事件：小车/货箱越界，或累计硬冲击 >= 3 次
    terminal_failure = 0.0
    if not _FAILED[0]:
        if ex_cart > 1.05 or ex_crate > 1.12 or _HIT_COUNT[0] >= 3:
            _FAILED[0] = True
            terminal_failure = -100.0

    # ------------------------------------------------------------------
    # 6) 动作代价 / 时间代价
    # ------------------------------------------------------------------
    action_cost = -0.0005 * (action[0] * action[0] + action[1] * action[1])
    time_cost = -0.002

    components = {
        "approach_cargo": approach_cargo,
        "progress": progress,
        "dock_enter": dock_enter,
        "roughness": roughness,
        "action_cost": action_cost,
        "time_cost": time_cost,
        "hard_hit": hard_hit,
        "settle_hold": settle_hold,
        "terminal_success": terminal_success,
        "terminal_failure": terminal_failure,
        "bounds_guard": bounds_guard,
    }

    total_reward = (
        approach_cargo
        + progress
        + dock_enter
        + roughness
        + action_cost
        + time_cost
        + hard_hit
        + settle_hold
        + terminal_success
        + terminal_failure
        + bounds_guard
    )

    return float(total_reward), components
```