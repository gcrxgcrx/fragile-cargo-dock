# =====================================================================
# 模块级状态（必须写在函数之前，用于一次性事件与回合边界检测）
# =====================================================================
_PREV_T = [-1.0]          # 上一帧 time_fraction（obs[18]），用于检测 episode 重置
_STREAK = [0]             # "泊位内 + 对齐 + 慢" 的连续步数
_PAID_SUCCESS = [False]   # terminal_success 一次性发放标记
_PAID_FAIL = [False]      # terminal_failure 一次性发放标记
_ENTERED = [False]        # dock_enter 一次性发放标记
_HIT_COUNT = [0]          # 本回合硬冲击次数（由接触+接近速度代理）
_HIT_ACTIVE = [0]         # 上一帧是否处于硬冲击（只统计上升沿，避免重复计数）

_HALF_W = 5.0             # 仓库半宽（米）：obs[12] 是"米 / 半宽"，obs[0]*5.0 为小车 x
_HALF_H = 4.0             # 仓库半高（米）：obs[13] 是"米 / 半高"，obs[1]*4.0 为小车 y
_K_ROUGH = 0.12           # roughness 代理系数（1/(m/s)）
_K_SETTLE = 0.5           # 停稳期每步收益
_HARD_CLOSING = 1.0       # 视为"硬冲击"的接近速度阈值 (m/s)
_COS30 = 0.8660254        # cos(30°)：朝向对齐判据


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -----------------------------------------------------------------
    # 0) 回合边界检测：obs[18] 在回合内单调递增，重置时回落
    # -----------------------------------------------------------------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID_SUCCESS[0] = False
        _PAID_FAIL[0] = False
        _ENTERED[0] = False
        _HIT_COUNT[0] = 0
        _HIT_ACTIVE[0] = 0
    _PREV_T[0] = t

    # -----------------------------------------------------------------
    # 1) approach_cargo: 小车→货箱距离本帧缩短量（米，有符号对称）
    #    车体系相对向量长度与旋转无关，可直接由 obs[6],obs[7] 还原
    # -----------------------------------------------------------------
    d_cc_prev = 3.0 * ((obs[6] * obs[6] + obs[7] * obs[7]) ** 0.5)
    d_cc_now = 3.0 * ((next_obs[6] * next_obs[6] + next_obs[7] * next_obs[7]) ** 0.5)
    approach_cargo = 1.0 * (d_cc_prev - d_cc_now)

    # -----------------------------------------------------------------
    # 2) progress: 货箱→泊位距离本帧缩短量（米，有符号对称，禁止 max(0,·)）
    # -----------------------------------------------------------------
    d_cd_prev = (((obs[12] * _HALF_W) ** 2 + (obs[13] * _HALF_H) ** 2)) ** 0.5
    d_cd_now = (((next_obs[12] * _HALF_W) ** 2 + (next_obs[13] * _HALF_H) ** 2)) ** 0.5
    progress = 1.0 * (d_cd_prev - d_cd_now)

    # -----------------------------------------------------------------
    # 3) dock_enter: 货箱首次"完全进入"泊位容差 —— 整局只发一次
    # -----------------------------------------------------------------
    in_dock = 0.0
    if abs(next_obs[12]) <= 0.024 and abs(next_obs[13]) <= 0.030:
        in_dock = 1.0
    dock_enter = 0.0
    if in_dock > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0

    # -----------------------------------------------------------------
    # 4) roughness: 接触冲量比例的观测代理
    #    closing = 小车前向速度 - 货箱沿车头方向速度（只在接触时罚）
    # -----------------------------------------------------------------
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    roughness = -_K_ROUGH * contact * closing
    # 量级说明：closing=1.0 m/s -> -0.12，明显大于正常推箱的单步 progress(≈0.02~0.05)；
    # closing≈0（匀速推箱，车与箱同速）-> 0，不误判正常推动为撞击。

    # -----------------------------------------------------------------
    # 5) hard_hit: 单步内一次硬冲击的固定惩罚（上升沿计入次数，供失败判定）
    # -----------------------------------------------------------------
    is_hard = 0.0
    if contact > 0.5 and closing > _HARD_CLOSING:
        is_hard = 1.0
    hard_hit = -0.5 * is_hard
    if is_hard > 0.5 and _HIT_ACTIVE[0] == 0:
        _HIT_COUNT[0] += 1
    if is_hard > 0.5:
        _HIT_ACTIVE[0] = 1
    else:
        _HIT_ACTIVE[0] = 0

    # -----------------------------------------------------------------
    # 6) action_cost / time_cost
    # -----------------------------------------------------------------
    action_cost = -0.0005 * (action[0] * action[0] + action[1] * action[1])
    time_cost = -0.002

    # -----------------------------------------------------------------
    # 7) 停稳期每步收益 + 一次性完成事件（两者共存，互不关闭）
    #    完成谓词与分项表一致：泊位内 + 朝向误差<30° + 货箱速度<0.05 m/s
    # -----------------------------------------------------------------
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    aligned = 1.0 if next_obs[10] >= _COS30 else 0.0
    settled = 0.0
    if in_dock > 0.5 and aligned > 0.5 and crate_speed < 0.05:
        settled = 1.0
    if settled > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0
    settle_hold = _K_SETTLE * settled          # 谓词成立就每步发，无任何关闭开关
    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID_SUCCESS[0]:
        _PAID_SUCCESS[0] = True
        terminal_success = 300.0               # 一次性；单步裁剪后约等于 +20
    # 注：停稳期每步收益与一次性事件并存。理由：只有一次性事件时，策略会在泊位外
    #     停住、永远不满足保持条件；只有每步收益时，策略会赖在停稳态不走完 10 步。
    #     强度比由"推进项是否足够强"控制（progress 为 ±1.0/m 的有符号增量）。

    # -----------------------------------------------------------------
    # 8) 越界守卫（小车由 obs[0],obs[1]；货箱由旋转还原的世界坐标）
    # -----------------------------------------------------------------
    ax = abs(next_obs[0])
    ay = abs(next_obs[1])
    bx = ax - 0.90
    if bx < 0.0:
        bx = 0.0
    by = ay - 0.90
    if by < 0.0:
        by = 0.0

    cart_xm = next_obs[0] * _HALF_W
    cart_ym = next_obs[1] * _HALF_H
    rx = next_obs[6] * 3.0
    ry = next_obs[7] * 3.0
    crate_xm = cart_xm + rx * next_obs[2] - ry * next_obs[3]
    crate_ym = cart_ym + rx * next_obs[3] + ry * next_obs[2]
    cbx = abs(crate_xm / _HALF_W) - 0.90
    if cbx < 0.0:
        cbx = 0.0
    cby = abs(crate_ym / _HALF_H) - 0.90
    if cby < 0.0:
        cby = 0.0

    out_of_bounds = 0.0
    out_of_bounds -= 6.0 * bx + 20.0 * bx * bx
    out_of_bounds -= 6.0 * by + 20.0 * by * by
    out_of_bounds -= 3.0 * cbx + 10.0 * cbx * cbx
    out_of_bounds -= 3.0 * cby + 10.0 * cby * cby
    # |x|=0.95 -> -0.35；|x|=1.00 -> -0.80；|x|=1.05 -> -1.35（远大于单步推进项）

    # -----------------------------------------------------------------
    # 9) terminal_failure: 越界 或 累计硬冲击 >= 3 —— 整局只发一次
    # -----------------------------------------------------------------
    failed = False
    if ax > 1.05 or ay > 1.05:
        failed = True
    if _HIT_COUNT[0] >= 3:
        failed = True
    terminal_failure = 0.0
    if failed and not _PAID_FAIL[0]:
        _PAID_FAIL[0] = True
        terminal_failure = -100.0

    # -----------------------------------------------------------------
    # 自检记录（估算，dt≈0.05s，推箱速度≈0.4~0.6 m/s）
    #   R_idle    ≈ -0.002                       （只有 time_cost）
    #   R_push    ≈ +0.02 ~ +0.05                （progress/approach 增量
    #                                             -0.002 -0.0005 - 极小的 roughness）
    #   R_settled ≈ +0.50                        （停稳每步 +0.5 -0.002）
    #   R_push - R_idle ≈ 0.02~0.05 > 所有常规罚项单步量级（≤0.012）  ✔
    #   R_settled > R_push                                            ✔
    #   自检① ②(推) > ①(静止)                                        ✔
    #   自检② ④(入坞停稳: +5 一次性 + 0.5*10 停稳 + ~19.5 事件 ≈ 29.5)
    #          > ③(泊位外 0.3m 悬停 400 步: -0.8)                     ✔
    #   自检③ ⑤(closing=1.0) 比 ⑥(closing=0.05) 低 ≈0.114 + 可能的
    #          hard_hit 0.5，与单步推进项(≈0.02~0.05)同量级或更大      ✔
    #   自检④ 同一停稳态连续调用 12 次，每步差值恒为 +0.5-0.002        ✔
    #   自检⑤ 中心(|x|=0) vs 边缘(|x|=1.05): 0 vs -1.35               ✔
    # -----------------------------------------------------------------
    total_reward = (approach_cargo + progress + dock_enter + roughness
                    + action_cost + time_cost + hard_hit + settle_hold
                    + out_of_bounds + terminal_success + terminal_failure)

    components = {
        "approach_cargo": approach_cargo,
        "progress": progress,
        "dock_enter": dock_enter,
        "roughness": roughness,
        "action_cost": action_cost,
        "time_cost": time_cost,
        "hard_hit": hard_hit,
        "settle_hold": settle_hold,
        "out_of_bounds": out_of_bounds,
        "terminal_success": terminal_success,
        "terminal_failure": terminal_failure,
    }
    return (float(total_reward), components)