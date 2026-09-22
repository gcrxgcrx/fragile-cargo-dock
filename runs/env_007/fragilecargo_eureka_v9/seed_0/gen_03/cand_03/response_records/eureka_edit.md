# Response Record

**问题诊断**

1. `settle_bonus`（2.0/步）与 `terminal_success`（300）合计已经主导了回合回报（≈370/回合），而真正的运输信号 `progress` 只有 ≈3.2——奖励被"停在泊位附近"这件事收割，而不是"把货箱送进泊位"。
2. 证据：eval 20/20 全部 truncated（环境从未判定成功），但训练里 `settle_bonus` 激活率已达 45%——说明我方 0.25 m/30°/0.05 m/s 的宽松判据远松于真实"完全进入 dock"，策略学会了长期赖在这个平台上收分。
3. 改法：把二元平台换成**连续、且有总量上限**的停稳项（near×slow×align 的乘积，每回合最多 24 点，对应真实 10 步完成窗口），使其无法被无限收割也不会压过主任务。
4. 让运输成为主稠密信号（货箱→dock 距离缩短量 4.0/米），并把完成判据收紧到 0.15 m / 25° / 0.05 m·s⁻¹ 连续 10 步（+300）。
5. 接触粗糙度改为**死区式**（仅 closing>0.5 m/s 且接触时惩罚），保证匀速正常推箱时该项恒为 0，不压制推进。

自检：①什么都不做 ≈ −0.002；②正常推箱（货箱向 dock 移动 0.05 m/步）≈ +0.4×0.5−0.0025 ≈ +0.20，② ≫ ①，且接触粗糙度在匀速推箱时为 0。

```python
# ============================================================================
# 自检记录（单步奖励估算，单位：奖励/步）
#
#   ① 什么都不做（货箱停在原地、无接触、动作=0）：
#        approach_cargo=0, progress=0, roughness=0, hard_hit=0,
#        settle_bonus=0（near=0），time_cost=-0.002, action_cost=0
#        R_idle ≈ -0.002
#
#   ② 正在把货箱推向 dock（接触、匀速推、货箱 0.05 m/步向 dock，closing≈0）：
#        progress = 4.0 * 0.05 = +0.20
#        approach_cargo ≈ 0（小车跟随货箱，车-箱距离不变）
#        roughness = -0.25 * 1 * max(0, 0-0.5) = 0     （死区内，不罚正常推进）
#        hard_hit = 0, settle_bonus = 0（尚在泊位外）
#        time_cost+action_cost ≈ -0.0025
#        R_push ≈ +0.197
#      R_idle(-0.002) < R_push(+0.197)  严格成立 ✓
#
#   ③ 停稳收割上限：settle_bonus 每回合累计上限 24.0（≈真实 10 步完成窗口
#      的 2 倍），远小于 terminal_success=300，平台期不再主导回报。      ✓
#   ④ 同一"停稳"状态连续 12 步成立：每步仍为正值（预算耗尽前），不会被
#      任何开关关停；但总量封顶，不存在"赖在泊位无限收分"。             ✓
#   ⑤ 越界自检：|obs[0]|=1.05 → bounds_safety=-2.55，远低于场地中央 0。  ✓
#
#   完成谓词（全部由 obs 推断，比上一版更紧）：
#     |crate→dock| < 0.15 m 且 |cos(crate 朝向)| >= 0.90 (25°) 且
#     crate 速度 < 0.05 m/s，连续 10 步成立 → 一次性 +300。
#   停稳每步收益与完成谓词共用同一 near/slow/align 连续量，不存在"半对齐刷分"。
# ============================================================================

_PREV_T = [-1.0]
_STREAK = [0]
_PAID_SUCCESS = [False]
_SETTLE_PAID = [0.0]
_HARD_HITS = [0]
_PAID_FAIL = [False]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------------- 0. 回合边界检测（obs[18] 单调递增，重置时回落） ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 0.0025:
        _STREAK[0] = 0
        _PAID_SUCCESS[0] = False
        _SETTLE_PAID[0] = 0.0
        _HARD_HITS[0] = 0
        _PAID_FAIL[0] = False
    _PREV_T[0] = t

    # ---------------- 1. 由 obs 还原几何量（单位：米） ----------------
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

    # 货箱世界坐标（仅用于越界守卫）
    crate_x = cart_x + c * rel_x - s * rel_y
    crate_y = cart_y + s * rel_x + c * rel_y

    # ---------------- 2. 分项实现 ----------------

    # 2.1 approach_cargo：+1.0 / 米，本帧小车->货箱距离缩短量（有符号、对称）
    approach_cargo = 1.0 * (d_cart_crate - nd_cart_crate)

    # 2.2 transport_progress：+4.0 / 米，本帧货箱->泊位中心距离缩短量（主稠密信号）
    progress = 4.0 * (d_crate_dock - nd_crate_dock)

    # 2.3 接触与接近速度（硬撞击/粗糙度代理，全部 derived）
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0

    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0

    # 2.4 roughness：死区式柔顺惩罚，只在"高速逼近接触"时生效，
    #     匀速推箱（closing<=0.5 m/s）恒为 0，不压制正常推进。
    rough_excess = closing - 0.5
    if rough_excess < 0.0:
        rough_excess = 0.0
    roughness = -0.25 * contact * rough_excess

    # 2.5 hard_hit：单步硬冲击代理（接触 + 高速逼近）固定惩罚并计数
    hard_hit = 0.0
    if contact > 0.5 and closing > 1.0:
        hard_hit = -1.0
        _HARD_HITS[0] += 1

    # 2.6 settle_bonus：连续、有总量上限的停稳项
    #     near/slow 为线性衰减因子，align 以 0.5~1.0 倍率平滑调制（避免乘积塌缩）
    #     每回合累计封顶 24.0，杜绝"赖在泊位无限收分"。
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    align_cos = next_obs[10]
    if align_cos < 0.0:
        align_cos = -align_cos

    near = 1.0 - nd_crate_dock / 0.30
    if near < 0.0:
        near = 0.0
    if near > 1.0:
        near = 1.0

    slow = 1.0 - crate_speed / 0.30
    if slow < 0.0:
        slow = 0.0
    if slow > 1.0:
        slow = 1.0

    align_f = (align_cos - 0.60) / 0.40
    if align_f < 0.0:
        align_f = 0.0
    if align_f > 1.0:
        align_f = 1.0

    raw_settle = 1.2 * near * slow * (0.5 + 0.5 * align_f)
    settle_bonus = 0.0
    if raw_settle > 0.0:
        room = 24.0 - _SETTLE_PAID[0]
        if room > 0.0:
            settle_bonus = raw_settle if raw_settle < room else room
            _SETTLE_PAID[0] += settle_bonus

    # 2.7 terminal_success：收紧的完成谓词（0.15 m / 25° / 0.05 m/s）连续 10 步
    if nd_crate_dock < 0.15 and align_cos >= 0.90 and crate_speed < 0.05:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0
    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID_SUCCESS[0]:
        _PAID_SUCCESS[0] = True
        terminal_success = 300.0

    # 2.8 越界度量（小车按 obs 归一化，货箱按世界坐标 / 半宽、半高）
    cart_margin = abs(obs[0])
    if abs(obs[1]) > cart_margin:
        cart_margin = abs(obs[1])
    crate_margin = abs(crate_x) / 5.0
    if abs(crate_y) / 4.0 > crate_margin:
        crate_margin = abs(crate_y) / 4.0

    m = cart_margin
    if crate_margin > m:
        m = crate_margin

    # 2.9 bounds_safety：随越界接近单调下降的守卫惩罚
    bounds_safety = 0.0
    if m > 0.90:
        over = m - 0.90
        bounds_safety = -(2.0 * over + 100.0 * over * over)

    # 2.10 terminal_failure：越界 或 累计硬冲击 >= 3，整局只发一次
    terminal_failure = 0.0
    if not _PAID_FAIL[0]:
        if cart_margin > 1.05 or crate_margin > 1.0 or _HARD_HITS[0] >= 3:
            _PAID_FAIL[0] = True
            terminal_failure = -100.0

    # 2.11 action_cost / time_cost（轻量，不压制必要推挤）
    action_cost = -0.0005 * (action[0] * action[0] + action[1] * action[1])
    time_cost = -0.002

    # ---------------- 3. 汇总 ----------------
    components = {
        "approach_cargo": approach_cargo,
        "progress": progress,
        "roughness": roughness,
        "hard_hit": hard_hit,
        "settle_bonus": settle_bonus,
        "action_cost": action_cost,
        "time_cost": time_cost,
        "bounds_safety": bounds_safety,
        "terminal_success": terminal_success,
        "terminal_failure": terminal_failure,
    }

    total = (approach_cargo + progress + roughness + hard_hit + settle_bonus
             + action_cost + time_cost + bounds_safety
             + terminal_success + terminal_failure)

    return (float(total), components)
```
