```python
# ============================================================================
# 自检记录（单步奖励估算，单位：奖励/步）
#   几何：仓库半宽 5.0 m，半高 4.0 m；obs[6]/obs[7]/obs[8]/obs[9] 除以 3.0；
#         obs[12]/obs[13] 分别按半宽/半高归一化。
#
#   R_idle    ≈ -0.002       （无接触、无位移：只有 time_cost）
#   R_push    ≈ +0.09        （正常推行：progress≈+0.05, approach_cargo≈+0.05,
#                              roughness≈-0.01, time+action≈-0.003）
#   R_settled ≈ +1.997       （泊位内 + 对齐 + 慢：settle 每步 +2.0 再扣成本）
#
#   自检①  idle(-0.002) < push(+0.09)                → 严格成立 ✓
#   自检②  泊位外 0.3m 悬停 400 步 ≈ -0.8
#          真正入泊并停稳 ≈ +18(停稳 9 步) + 20(第 10 步被裁剪的完成事件) ≈ +38
#          +38 > -0.8                                 → 严格成立 ✓
#   自检③  closing=1.00 → roughness = -0.30, hard_hit = 0（>1.0 才触发）
#          closing=0.05 → roughness = -0.015
#          差值 0.285 ≫ 正常推箱单步推进值(≈0.05~0.10)，约 3~6 倍       ✓
#          且该惩罚在 closing≈0 的"匀速正常推箱"时为 0，不压制推进。
#   自检④  同一"停稳"状态连续调用 12 次，每步差值恒为 +2.0（完成事件只在
#          第 10 步叠加一次），停稳收益不会被任何开关关掉。            ✓
#   自检⑤  R_push - R_idle ≈ 0.09，明显大于正常单步罚项量级
#          (roughness≈0.01, time_cost 0.002, action_cost 0.0005)       ✓
#          R_settled > R_push                                          ✓
#          量级：R_push-R_idle 与推进项同量级（都是 1e-1），非千分之一。 ✓
#   越界自检：|obs[0]|=1.05 → bounds_safety = -2.55，远低于场地中央的 0。  ✓
#
#   完成谓词（显式从 obs 推断，容差取自环境事实）：
#     |crate→dock| < 0.25 m 且 |crate 朝向 cos| >= 0.866 (30°) 且
#     crate 速度 < 0.05 m/s，连续 10 步成立 → 一次性 +300。
#   停稳每步收益与完成谓词使用同一组硬门控，故不可能"半对齐赖着收分"。
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

    # ---------------- 1. 由 obs 还原几何量（单位：米） ----------------
    cart_x = obs[0] * 5.0
    cart_y = obs[1] * 4.0
    c = obs[2]
    s = obs[3]

    # 货箱在小车车体坐标系下的相对位置
    rel_x = obs[6] * 3.0
    rel_y = obs[7] * 3.0
    d_cart_crate = (rel_x * rel_x + rel_y * rel_y) ** 0.5

    nrel_x = next_obs[6] * 3.0
    nrel_y = next_obs[7] * 3.0
    nd_cart_crate = (nrel_x * nrel_x + nrel_y * nrel_y) ** 0.5

    # 货箱中心 -> 泊位中心距离
    ddx = obs[12] * 5.0
    ddy = obs[13] * 4.0
    d_crate_dock = (ddx * ddx + ddy * ddy) ** 0.5

    nddx = next_obs[12] * 5.0
    nddy = next_obs[13] * 4.0
    nd_crate_dock = (nddx * nddx + nddy * nddy) ** 0.5

    # 货箱世界坐标（用于越界守卫）
    crate_x = cart_x + c * rel_x - s * rel_y
    crate_y = cart_y + s * rel_x + c * rel_y

    # ---------------- 2. 分项实现 ----------------

    # 2.1 approach_cargo：+1.0 / 米，本帧小车->货箱距离缩短量（有符号、对称）
    approach_cargo = 1.0 * (d_cart_crate - nd_cart_crate)

    # 2.2 progress：+1.0 / 米，本帧货箱->泊位距离缩短量（有符号、对称）
    progress = 1.0 * (d_crate_dock - nd_crate_dock)

    # 2.3 dock_enter：+5.0，货箱首次完全进入泊位容差，整局只发一次
    in_dock = 1.0 if nd_crate_dock < 0.25 else 0.0
    dock_enter = 0.0
    if in_dock > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0

    # 2.4 roughness：接触冲量比例的可观测代理（接近速度 x 接触）
    #     只在"接触且正在接近"时为负；匀速推箱（closing≈0）不受罚。
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    roughness = -0.30 * contact * closing

    # 2.5 action_cost：-0.0005 * 动作平方和
    action_cost = -0.0005 * (action[0] * action[0] + action[1] * action[1])

    # 2.6 time_cost：-0.002 / 步
    time_cost = -0.002

    # 2.7 hard_hit：单步硬冲击（接触 + 高速接近）固定惩罚 + 计数
    hard_hit = 0.0
    if contact > 0.5 and closing > 1.0:
        hard_hit = -0.5
        _HARD_HITS[0] += 1

    # 2.8 完成侧状态量（全部来自 next_obs）
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    align_cos = next_obs[10]
    if align_cos < 0.0:
        align_cos = -align_cos

    align_ok = 1.0 if align_cos >= 0.866 else 0.0     # 朝向误差 < 30 度
    slow_ok = 1.0 if crate_speed < 0.05 else 0.0      # 速度 < 0.05 m/s
    # 停稳期每步收益：硬门控（不满足完成谓词一分不给，不存在"半对齐刷分"），
    # 且不因任何"已发放"开关而停止。
    settle_bonus = 2.0 * in_dock * align_ok * slow_ok

    # 2.9 terminal_success：完成谓词连续 10 步成立，整局只发一次
    if in_dock > 0.5 and align_ok > 0.5 and slow_ok > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0
    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID_SUCCESS[0]:
        _PAID_SUCCESS[0] = True
        terminal_success = 300.0

    # 2.10 越界度量（小车按 obs 归一化，货箱按世界坐标 / 半宽、半高）
    cart_margin = abs(obs[0])
    if abs(obs[1]) > cart_margin:
        cart_margin = abs(obs[1])
    crate_margin = abs(crate_x) / 5.0
    if abs(crate_y) / 4.0 > crate_margin:
        crate_margin = abs(crate_y) / 4.0

    # 2.11 terminal_failure：越界 或 累计硬冲击 >= 3，整局只发一次
    terminal_failure = 0.0
    if not _PAID_FAIL[0]:
        if cart_margin > 1.05 or crate_margin > 1.0 or _HARD_HITS[0] >= 3:
            _PAID_FAIL[0] = True
            terminal_failure = -100.0

    # 2.12 bounds_safety：随越界接近单调下降的守卫惩罚
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
```