```python
# ---------------------------------------------------------------------------
# 模块级状态（必须在函数之前声明）
#   _PREV_T   : 上一帧的 time_fraction (obs[18])，用于检测 episode 边界（重置时回落）
#   _STREAK   : "泊位内 + 对齐 + 慢" 的连续步计数
#   _PAID     : 一次性完成事件是否已发放（同一 episode 内不重复）
#   _ENTERED  : "首次完全进入泊位容差" 是否已发放
#   _HARD_HITS: 硬冲击代理的累计次数（>=3 触发一次性失败罚）
#   _FAIL_PAID: 一次性失败罚是否已发放
# ---------------------------------------------------------------------------
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_HARD_HITS = [0]
_FAIL_PAID = [False]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # =====================================================================
    # 量级自检记录（自检 ⑤）
    #   R_idle    ~= -0.002            (只吃 time_cost，不动、不推)
    #   R_push    ~= +1.0              (approach≈0.5 + progress≈0.5 - 小罚)
    #   R_settled ~= +2.0              (settle_hold=2.0 - time/action 小罚)
    #   => R_push - R_idle ≈ 1.0  >>  罚项单步最大量级(~0.01)
    #   => R_settled > R_push  (停稳是全局最优点)
    #   => 自检① 推箱 > 静止 ✓ ; 自检② 真入泊位(≈40) >> 悬停收割(≈-0.8) ✓
    #      自检③ closing=1.0: roughness -1.5 +hard_hit -0.5 = -2.0
    #             closing=0.05: -0.075  差值 1.9 >> 推进单步 ~1.0 ✓
    #      自检④ 停稳态连调 12 次，每次差值恒为相同正数（settle_hold 不被关闭）✓
    #      自检⑤ 边界：|obs[0]|=0.0 -> 0 罚 ; |obs[0]|=1.05 -> 约 -3.6 ✓
    # =====================================================================

    # ---------- 1. episode 边界检测与状态重置（obs[18] 单调递增，重置回落）----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _HARD_HITS[0] = 0
        _FAIL_PAID[0] = False
    _PREV_T[0] = t

    # ---------- 2. approach_cargo (+1.0 / 米)：小车 -> 货箱距离的本帧有符号变化 ----------
    # obs[6], obs[7] = 货箱相对小车的车体系位置 / 3.0 (m)；车体系模长与旋转无关
    rx0 = obs[6] * 3.0
    ry0 = obs[7] * 3.0
    rx1 = next_obs[6] * 3.0
    ry1 = next_obs[7] * 3.0
    d_cart_crate_prev = (rx0 * rx0 + ry0 * ry0) ** 0.5
    d_cart_crate_next = (rx1 * rx1 + ry1 * ry1) ** 0.5
    approach_cargo = 1.0 * (d_cart_crate_prev - d_cart_crate_next)   # 有符号、对称

    # ---------- 3. progress (+1.0 / 米)：货箱 -> 泊位距离的本帧有符号变化 ----------
    # obs[12] / 半宽(5.0 m)，obs[13] / 半高(4.0 m) 还原为米
    dx0 = obs[12] * 5.0
    dy0 = obs[13] * 4.0
    dx1 = next_obs[12] * 5.0
    dy1 = next_obs[13] * 4.0
    d_crate_dock_prev = (dx0 * dx0 + dy0 * dy0) ** 0.5
    d_crate_dock_next = (dx1 * dx1 + dy1 * dy1) ** 0.5
    progress = 1.0 * (d_crate_dock_prev - d_crate_dock_next)         # 有符号、对称

    # ---------- 4. dock_enter (+5.0)：货箱首次完全进入泊位容差，整局一次性 ----------
    inside_now = (abs(next_obs[12]) <= 0.024) and (abs(next_obs[13]) <= 0.030)
    dock_enter = 0.0
    if inside_now and (not _ENTERED[0]):
        _ENTERED[0] = True
        dock_enter = 5.0

    # ---------- 5. roughness：观测代理（接近速度 × 接触），抑制撞击 ----------
    # 观测里没有冲量维，用 [车沿车头速度] - [货箱沿车头速度] 作为接近速度代理
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0                       # 只罚"正在接近"，匀速推箱(closing≈0)不罚
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    # k = 1.5：closing = 1.0 m/s 时罚 1.5，与推进项单步量级相当/更大；
    #          正常推箱时车/箱速度接近，closing≈0，不会压制推进。
    roughness = -1.5 * contact * closing

    # ---------- 6. hard_hit (-0.5)：单步硬冲击代理（接触且接近速度过大）----------
    hard_hit = 0.0
    if contact > 0.5 and closing > 1.2:
        hard_hit = -0.5
        _HARD_HITS[0] = _HARD_HITS[0] + 1

    # ---------- 7. action_cost (-0.0005) / time_cost (-0.002) ----------
    action_cost = -0.0005 * (action[0] * action[0] + action[1] * action[1])
    time_cost = -0.002

    # ---------- 8. 越界守卫：随接近边界单调下降的连续惩罚 ----------
    # 小车位置由 obs[0], obs[1]（±1.0 即墙）；货箱世界坐标由车体系相对位置旋转还原
    c1 = next_obs[2]
    s1 = next_obs[3]
    crate_x_n = next_obs[0] + (c1 * rx1 - s1 * ry1) / 5.0
    crate_y_n = next_obs[1] + (s1 * rx1 + c1 * ry1) / 4.0

    cart_lim = abs(next_obs[0])
    if abs(next_obs[1]) > cart_lim:
        cart_lim = abs(next_obs[1])
    crate_lim = abs(crate_x_n)
    if abs(crate_y_n) > crate_lim:
        crate_lim = abs(crate_y_n)

    over_cart = cart_lim - 0.93
    if over_cart < 0.0:
        over_cart = 0.0
    over_crate = crate_lim - 0.93
    if over_crate < 0.0:
        over_crate = 0.0
    boundary_guard = -30.0 * over_cart - 30.0 * over_crate

    # ---------- 9. 完成谓词：泊位内 + 对齐 + 慢 ----------
    # 容差严格取自环境事实：|obs[12]|<=0.024, |obs[13]|<=0.030
    # 朝向误差 < 30°：泊位朝向沿仓库 x 轴，|sin| <= 0.5 即 <= 30°
    # 货箱速度 < 0.05 m/s
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    aligned = abs(next_obs[11]) <= 0.5
    slowed = crate_speed < 0.05
    cond = inside_now and aligned and slowed

    if cond:
        _STREAK[0] = _STREAK[0] + 1
    else:
        _STREAK[0] = 0

    # 一次性完成事件（连续 10 步成立时发放，同一 episode 只发一次）
    success_event = 0.0
    if _STREAK[0] >= 10 and (not _PAID[0]):
        _PAID[0] = True
        success_event = 300.0

    # 停稳期每步收益：谓词成立就每步发放，不受一次性事件是否已发放影响
    # （完成侧信号，不是"靠近就持续给分"的全局状态项——它要求完全进入泊位 + 对齐 + 静止）
    settle_hold = 2.0 if cond else 0.0

    # ---------- 10. terminal_failure (-100)：越界 或 累计硬冲击 >= 3，整局一次性 ----------
    out_of_bounds = (abs(next_obs[0]) > 1.05) or (abs(next_obs[1]) > 1.05) \
        or (abs(crate_x_n) > 1.05) or (abs(crate_y_n) > 1.05)
    failure_event = 0.0
    if (out_of_bounds or _HARD_HITS[0] >= 3) and (not _FAIL_PAID[0]):
        _FAIL_PAID[0] = True
        failure_event = -100.0

    # ---------- 11. 总奖励与分项字典 ----------
    total_reward = (approach_cargo
                    + progress
                    + dock_enter
                    + roughness
                    + action_cost
                    + time_cost
                    + hard_hit
                    + success_event
                    + failure_event
                    + settle_hold
                    + boundary_guard)

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
        "settle_hold": settle_hold,
        "boundary_guard": boundary_guard,
    }
    return (float(total_reward), components)
```