```python
# =====================================================================
# 模块级状态
#   - 签名里没有 done，用 obs[18]（时间预算比例，回合内单调递增、重置时回落）
#     检测回合边界并重置状态，避免跨 episode 污染。
#   - 一次性事件：dock_enter(+5) / terminal_success(+300 连续10步) /
#     terminal_failure(-100)；_HITS 为自维护的硬冲击计数代理。
# =====================================================================
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_FAILED = [False]
_HITS = [0]
_PREV_CONTACT = [0.0]
_PREV_CLOSING = [0.0]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -----------------------------------------------------------------
    # 0) 回合边界检测与状态重置
    # -----------------------------------------------------------------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _FAILED[0] = False
        _HITS[0] = 0
        _PREV_CONTACT[0] = 0.0
        _PREV_CLOSING[0] = 0.0
    _PREV_T[0] = t

    # -----------------------------------------------------------------
    # 1) 距离还原（米）
    #    小车→货箱：obs[6],obs[7] 是车体系相对位移 /3.0 m（旋转不变，取模即距离）
    #    货箱→泊位：obs[12],obs[13] 是有符号偏移 /半宽(5.0)、/半高(4.0)
    # -----------------------------------------------------------------
    ccx = obs[6] * 3.0
    ccy = obs[7] * 3.0
    nccx = next_obs[6] * 3.0
    nccy = next_obs[7] * 3.0
    d_cc = (ccx * ccx + ccy * ccy) ** 0.5
    d_cc_n = (nccx * nccx + nccy * nccy) ** 0.5

    ddx = obs[12] * 5.0
    ddy = obs[13] * 4.0
    nddx = next_obs[12] * 5.0
    nddy = next_obs[13] * 4.0
    d_dock = (ddx * ddx + ddy * ddy) ** 0.5
    d_dock_n = (nddx * nddx + nddy * nddy) ** 0.5

    # 势函数形式，有符号、对称：本帧更近给正分，更远给负分（禁止 max(0,·)）
    approach_cargo = d_cc - d_cc_n          # +1.0 / 米
    progress = d_dock - d_dock_n            # +1.0 / 米

    # -----------------------------------------------------------------
    # 2) roughness：接触冲量比例的可观测代理（-0.02/(N·s) 的观测化实现）
    #    closing = 小车前向速度 - 货箱速度沿车头方向的分量（仅在接触时起作用）
    #    k = 0.5：closing=1.0 m/s 时惩罚 0.5，与正常推进的单步量级相当/更大；
    #    接触但未接近（closing≈0，匀速推箱）不产生惩罚。
    # -----------------------------------------------------------------
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    if closing > 3.0:
        closing = 3.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    roughness = -0.5 * contact * closing    # 轻拿轻放：只在"接触且正在接近"时罚

    # -----------------------------------------------------------------
    # 3) hard_hit：单步硬冲击（接触上升沿或接触中接近速度重新越过 1.0 m/s）
    #    固定 -0.5，并累加到自维护计数；去抖避免一次冲击被重复计数。
    # -----------------------------------------------------------------
    hard_hit = 0.0
    if contact > 0.5 and closing >= 1.0 and (_PREV_CONTACT[0] < 0.5 or _PREV_CLOSING[0] < 1.0):
        hard_hit = -0.5
        _HITS[0] += 1
    _PREV_CONTACT[0] = contact
    _PREV_CLOSING[0] = closing

    # -----------------------------------------------------------------
    # 4) action_cost / time_cost
    # -----------------------------------------------------------------
    a0 = action[0]
    a1 = action[1]
    action_cost = -0.0005 * (a0 * a0 + a1 * a1)
    time_cost = -0.002

    # -----------------------------------------------------------------
    # 5) 完成谓词（容差完全取自环境事实，不得放宽）
    #    完全在泊位内：|obs[12]| <= 0.024 且 |obs[13]| <= 0.030
    #    朝向对齐：|atan2(obs[11],obs[10])| < 30deg  <=>  obs[10] >= 0.866
    #    几乎静止：货箱世界速度 < 0.05 m/s
    # -----------------------------------------------------------------
    ax12 = next_obs[12]
    if ax12 < 0.0:
        ax12 = -ax12
    ay13 = next_obs[13]
    if ay13 < 0.0:
        ay13 = -ay13
    inside_dock = (ax12 <= 0.024) and (ay13 <= 0.030)
    aligned = next_obs[10] >= 0.866
    cvx = next_obs[8] * 3.0
    cvy = next_obs[9] * 3.0
    crate_speed_sq = cvx * cvx + cvy * cvy
    slow = crate_speed_sq <= 0.0025          # 0.05 m/s 的平方
    docked = inside_dock and aligned and slow

    # dock_enter：首次完全进入容差，整局一次 +5.0
    dock_enter = 0.0
    if inside_dock and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0

    # terminal_success：完成谓词连续 10 步，整局一次 +300.0
    if docked:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0
    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        terminal_success = 300.0

    # 停稳期每步收益：谓词成立就每步发放，绝不被"一次性事件已发放"关掉。
    # 它只在与环境完成判据同口径的状态下生效，因此天然最多发 10 步
    # （第 10 步环境即终止 episode），不存在"赖着不动刷分"的空间。
    settled_bonus = 2.0 if docked else 0.0

    # -----------------------------------------------------------------
    # 6) 越界守卫：小车 obs[0]/obs[1]（+1.0 即墙），货箱世界坐标由
    #    obs[6],obs[7]（车体系，米）经姿态旋转 + 小车世界位置还原。
    #    |·| > 0.90 起单调下降惩罚，> 1.05 判越界。
    # -----------------------------------------------------------------
    cax = next_obs[0]
    if cax < 0.0:
        cax = -cax
    cay = next_obs[1]
    if cay < 0.0:
        cay = -cay

    ch = next_obs[2]
    sh = next_obs[3]
    rx = next_obs[6] * 3.0
    ry = next_obs[7] * 3.0
    crate_wx = next_obs[0] * 5.0 + (rx * ch - ry * sh)
    crate_wy = next_obs[1] * 4.0 + (rx * sh + ry * ch)
    cnx = crate_wx / 5.0
    cny = crate_wy / 4.0
    if cnx < 0.0:
        cnx = -cnx
    if cny < 0.0:
        cny = -cny

    over = 0.0
    if cax > 0.90:
        over += (cax - 0.90)
    if cay > 0.90:
        over += (cay - 0.90)
    if cnx > 0.90:
        over += (cnx - 0.90)
    if cny > 0.90:
        over += (cny - 0.90)
    bounds_guard = -5.0 * over

    # terminal_failure：小车/货箱越界，或累计硬冲击 >= 3，整局一次 -100.0
    failed_now = (cax > 1.05) or (cay > 1.05) or (cnx > 1.05) or (cny > 1.05) or (_HITS[0] >= 3)
    terminal_failure = 0.0
    if failed_now and not _FAILED[0]:
        _FAILED[0] = True
        terminal_failure = -100.0

    # -----------------------------------------------------------------
    # 自检记录（每步平均奖励，量级以"正常推箱时单步推进 ≈ 0.01~0.05"为基准）
    #   R_idle    ≈ -0.002                        （只有 time_cost）
    #   R_push    ≈ +0.01 ~ +0.05                 （progress/approach - 0.002 - 0.0005）
    #               -> R_push > R_idle，且推进项不被罚项淹没
    #   R_settled ≈ +2.0 - 0.002 - 0.0005 ≈ 1.997 （R_settled > R_push，停稳是全局最优点）
    # 自检③：k=0.5，closing=1.0 -> roughness -0.50；closing=0.05 -> -0.025；
    #        差值 0.475 >= 1.0 x（正常推箱单步推进 ~0.02）
    # 自检②：停泊位外 0.3 m 悬停 400 步 ≈ -0.8；
    #        真正入坞停稳 10 步 ≈ 10*2.0 + 5 + 300 > 0
    # 自检④：同一停稳状态连续调用 12 次，每次差值恒为 +2.0-(动作代价+时间代价) > 0
    # 自检⑤（越界）：场地中心 0 vs |obs[0]|=1.05 -> -0.75 且触发 -100（一次性）
    # -----------------------------------------------------------------
    total = (approach_cargo + progress + dock_enter + roughness + action_cost + time_cost
             + hard_hit + terminal_success + terminal_failure + settled_bonus + bounds_guard)

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
        "settled_bonus": settled_bonus,
        "bounds_guard": bounds_guard,
    }
    return (float(total), components)
```