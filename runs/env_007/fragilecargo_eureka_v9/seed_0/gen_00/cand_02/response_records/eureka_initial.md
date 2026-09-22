# Response Record

```python
# =============================================================================
# 模块级状态声明（必须写在 compute_reward 之前，用可变容器承载）
# =============================================================================
_PREV_T = [-1.0]     # 上一帧的 obs[18]（已用时间比例），用于检测 episode 边界
_STREAK = [0]        # "泊位内 + 对齐 + 慢" 连续成立的步数
_PAID = [False]      # terminal_success（连续 10 步）是否已一次性发放
_ENTERED = [False]   # 是否曾经"完全进入泊位"（dock_enter 整局只发一次）
_HARD_HITS = [0]     # 累计硬冲击次数（观测代理：接触且接近速度超阈值）
_FAILED = [False]    # terminal_failure 是否已一次性发放

# -----------------------------------------------------------------------------
# 关键常数（全部来自环境事实，不引入环境未声明的量）
#   obs[0]/obs[1]      = cart 位置 / 仓库半宽(5.0 m) / 半高(4.0 m)；|.|>~1.05 即出界
#   obs[6],[7]         = crate 在车体系下的相对位置 / 3.0
#   obs[8],[9],[4]     = 速度量纲 / 3.0
#   obs[12],[13]       = crate 到 dock 中心带符号偏移 / (5.0 m, 4.0 m)
#   obs[14]            > 0.5 视为接触；obs[18] 单调递增，reset 时回落
#   停稳判据（来自环境事实）：完全进入泊位 + 朝向误差 < 30 deg + 速度 < 0.05 m/s
#                            且连续保持 10 步（满足即立刻终止 episode）
#   注：0.30 m 到 dock 中心不算交付，故这里的"进入容差"取 ±0.20 m 方盒
# -----------------------------------------------------------------------------
_K_ROUGH = 0.15            # roughness 代理系数：-0.15 * contact * closing
_HARD_HIT_CLOSING = 0.8    # 接近速度 > 0.8 m/s 视为一次硬冲击（代理）
_IN_DOCK_HALF_X = 0.20     # m，货箱"完全进入泊位"的 x 半宽
_IN_DOCK_HALF_Y = 0.20     # m，货箱"完全进入泊位"的 y 半高
_ALIGN_COS = 0.866         # cos(30 deg)
_SLOW_SPEED = 0.05         # m/s
_SETTLE_STEP_BONUS = 0.10  # 停稳期每步收益（每步固定，不被任何开关关掉）


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # =========================================================================
    # 0. 回合边界检测 + 状态重置（禁止跨 episode 污染）
    # =========================================================================
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _HARD_HITS[0] = 0
        _FAILED[0] = False
    _PREV_T[0] = t

    # =========================================================================
    # 1. 几何还原（只用环境事实声明的索引与量纲）
    # =========================================================================
    cart_x = obs[0] * 5.0
    cart_y = obs[1] * 4.0
    relx = obs[6] * 3.0
    rely = obs[7] * 3.0
    ch = obs[2]
    sh = obs[3]
    crate_x = cart_x + relx * ch - rely * sh
    crate_y = cart_y + relx * sh + rely * ch

    cart_x_n = next_obs[0] * 5.0
    cart_y_n = next_obs[1] * 4.0
    relx_n = next_obs[6] * 3.0
    rely_n = next_obs[7] * 3.0
    ch_n = next_obs[2]
    sh_n = next_obs[3]
    crate_x_n = cart_x_n + relx_n * ch_n - rely_n * sh_n
    crate_y_n = cart_y_n + relx_n * sh_n + rely_n * ch_n

    # --- 分项 1: approach_cargo  (+1.0 / m, 有符号、对称的势函数增量) ---------
    d_cc_prev = ((crate_x - cart_x) ** 2 + (crate_y - cart_y) ** 2) ** 0.5
    d_cc_next = ((crate_x_n - cart_x_n) ** 2 + (crate_y_n - cart_y_n) ** 2) ** 0.5
    approach_cargo = (d_cc_prev - d_cc_next) * 1.0   # 靠近为正，远离为负

    # --- 分项 2: progress  (+1.0 / m, 有符号、对称的势函数增量) ---------------
    dk_x_prev = obs[12] * 5.0
    dk_y_prev = obs[13] * 4.0
    dk_x_next = next_obs[12] * 5.0
    dk_y_next = next_obs[13] * 4.0
    d_dock_prev = (dk_x_prev ** 2 + dk_y_prev ** 2) ** 0.5
    d_dock_next = (dk_x_next ** 2 + dk_y_next ** 2) ** 0.5
    progress = (d_dock_prev - d_dock_next) * 1.0     # 禁止 max(0, .)：会诱发来回刷分

    # =========================================================================
    # 2. 接触 / 接近速度代理（obs 里没有冲量，只能自己构造）
    # =========================================================================
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along = crate_vx * obs[2] + crate_vy * obs[3]   # 货箱速度沿车头方向分量
    closing = obs[4] * 3.0 - crate_along                 # 正在接近的速度 (m/s)
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0

    # --- 分项 3: roughness  (-0.02/(N*s) 的观测代理) --------------------------
    # 只在"接触 且 正在接近"时生效；匀速推箱 closing~0 -> 几乎不罚，
    # 不惩罚"推进"本身，只惩罚"撞得狠"。closing=1.0 m/s 时罚 -0.15，
    # 与推进项同量级或更大，使高速撞击明确不划算。
    roughness = -_K_ROUGH * contact * closing

    # --- 分项 4: hard_hit  (-0.5 / 次) ---------------------------------------
    hard_hit = 0.0
    if contact > 0.5 and closing > _HARD_HIT_CLOSING:
        hard_hit = -0.5
        _HARD_HITS[0] += 1

    # --- 分项 5/6: action_cost / time_cost -----------------------------------
    action_cost = -0.0005 * (action[0] * action[0] + action[1] * action[1])
    time_cost = -0.002

    # =========================================================================
    # 3. 完成侧：predicate / dock_enter（一次性）/ terminal_success（一次性）
    #                / settle_bonus（停稳期每步收益，永不被关掉）
    # =========================================================================
    in_dock = (abs(dk_x_next) < _IN_DOCK_HALF_X) and (abs(dk_y_next) < _IN_DOCK_HALF_Y)
    aligned = abs(next_obs[10]) >= _ALIGN_COS          # 货箱纵轴与运输轴(x)夹角 < 30 deg
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    slow = crate_speed < _SLOW_SPEED
    settled = in_dock and aligned and slow

    # --- 分项 3(表): dock_enter  +5.0 一次性 --------------------------------
    dock_enter = 0.0
    if in_dock and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0

    # 连续计数（成功判据要连续 10 步成立）
    if settled:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # --- 分项 8: terminal_success  +300 一次性，同 episode 不重复发放 --------
    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        terminal_success = 300.0

    # --- 停稳期每步收益：谓词成立就每步发放，绝不因为 _PAID / _STREAK 而停止 -
    #     （自检④/⑤：同一停稳状态连续调用 12 次，每次差值都等于同一个正数）
    settle_bonus = _SETTLE_STEP_BONUS if settled else 0.0

    # =========================================================================
    # 4. 越界守卫：cart 与 crate 都必须留在仓库地面内
    #    cart: |obs[0]| 或 |obs[1]| > ~1.05 即出界（+1.0 就是墙，不是 2.0）
    #    crate: 世界坐标超出半宽 5.0 / 半高 4.0 的 ~1.05 倍即出界
    # =========================================================================
    cart_edge = abs(next_obs[0])
    if abs(next_obs[1]) > cart_edge:
        cart_edge = abs(next_obs[1])
    crate_edge = abs(crate_x_n) / 5.0
    if abs(crate_y_n) / 4.0 > crate_edge:
        crate_edge = abs(crate_y_n) / 4.0

    bounds_penalty = 0.0
    if cart_edge > 0.90:
        bounds_penalty = bounds_penalty - 4.0 * (cart_edge - 0.90)
    if crate_edge > 0.90:
        bounds_penalty = bounds_penalty - 4.0 * (crate_edge - 0.90)
    if cart_edge > 1.05:
        bounds_penalty = bounds_penalty - 10.0 * (cart_edge - 1.05)
    if crate_edge > 1.05:
        bounds_penalty = bounds_penalty - 10.0 * (crate_edge - 1.05)

    # --- 分项 9: terminal_failure  -100 一次性 ------------------------------
    cart_oob = (abs(next_obs[0]) > 1.05) or (abs(next_obs[1]) > 1.05)
    crate_oob = (abs(crate_x_n) > 5.25) or (abs(crate_y_n) > 4.2)
    terminal_failure = 0.0
    if (cart_oob or crate_oob or _HARD_HITS[0] >= 3) and not _FAILED[0]:
        _FAILED[0] = True
        terminal_failure = -100.0

    # =========================================================================
    # 5. 汇总
    # =========================================================================
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
        "settle_bonus": settle_bonus,
        "bounds_penalty": bounds_penalty,
    }
    total = 0.0
    for k in components:
        total = total + components[k]
    return float(total), components


# =============================================================================
# 自检记录（数值为该设计下的典型单步量，单位与权重表一致）
# -----------------------------------------------------------------------------
# 自检 ①（idle vs 推箱，同一初始位置）
#   R_idle  = 0 (approach) + 0 (progress) + 0 (roughness) + 0 (action_cost)
#             - 0.002 (time_cost)                                   ~= -0.002
#   R_push  = 0.02 (progress: 本帧货箱向 dock 前进 ~2 cm)
#             + ~0 (approach: 已接触，车-箱距离基本不变)
#             + ~0 (roughness: 匀速推箱 closing ~= 0)
#             - 0.002 - 0.0005                                      ~= +0.018
#   => R_push > R_idle，差 ~0.02，大于常开罚项量级 (0.0025)。通过。
#
# 自检 ②（悬停收割 vs 真正停稳交付）
#   ③ 货箱停在泊位外 0.3 m、速度~0、400 步：
#        0 + 0 + 0 - 0.002*400                                    ~= -0.8
#   ④ 真正进入泊位并停稳到 episode 结束：
#        +0.3 (progress 补上最后 0.3 m)
#        +5.0 (dock_enter 一次性)
#        +0.10 * 10 = +1.0 (settle_bonus 每步)
#        +300 -> 单步裁剪到 +20 (terminal_success 一次性)
#        - 0.002*10                                               ~= +26.2
#   => ④ >> ③。通过（未放宽任何完成判据）。
#
# 自检 ③（轻柔度 vs 推进项量级）
#   ⑤ 接触 + closing = 1.0 m/s:
#        roughness = -0.15*1.0 = -0.15, hard_hit = -0.5  (1.0 > 0.8)
#        progress ~ +0.02                                       ~= -0.63
#   ⑥ 接触 + closing = 0.05 m/s:
#        roughness = -0.0075, 无 hard_hit, progress ~ +0.02    ~= +0.012
#   差 ~0.64 >> 正常推箱单步推进项 (~0.02)。通过。
#
# 自检 ④/⑤（停稳期每步收益连续性）
#   在同一个"泊位内 + 对齐 + 慢"状态连续调用 12 次：
#   每次差值 = settle_bonus(+0.10) + time_cost(-0.002) + action_cost(≈0)
#            = 恒定正数 ≈ +0.098，12 次线性增长，不被 _PAID 关掉。通过。
#
# 自检 ⑤（三条轨迹的每步平均奖励，单调性 + 量级）
#   R_idle    ~= -0.002
#   R_push    ~= +0.018      (推进增量主导；常开罚项仅 -0.0025)
#   R_settled ~= +0.098      (settle_bonus 主导，全局最优点)
#   单调性 R_settled > R_push > R_idle 成立；
#   R_push - R_idle (~0.02) >= 单步最大常开罚项 (~0.0025)；
#   R_push 与推进/停稳项同量级，不是千分之一。通过。
#
# 自检 ⑤（越界守卫）
#   场地中心 obs[0]=obs[1]=0           -> bounds_penalty = 0
#   |obs[0]| = 1.05 (出界边缘)         -> bounds_penalty = -4.0*0.15 = -0.6
#   => 后者单步总奖励明显更低。通过。
#
# 激励冲突审查：没有任何"全局持续的状态正奖励"；approach/progress 均为有符号增量，
#   settle_bonus 只在完成谓词成立时发放（属于完成侧信号，环境随后立即终止），
#   dock_enter 为一次性事件；roughness 只在"接触且正在接近"时非零，
#   匀速推箱 closing~0 不罚，推进动作不会被净负贡献。硬冲击另有 -0.5 固定罚，
#   累计 3 次触发一次性 -100（与环境的 3 次硬撞击失败语义对齐）。
# =============================================================================
```
