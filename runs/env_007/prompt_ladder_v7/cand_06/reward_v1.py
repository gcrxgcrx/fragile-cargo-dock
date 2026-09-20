# 模块级状态：用于回合边界检测、连续满足计数、一次性事件与首次进入
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]

# 常量（来自环境事实）
# 泊位容差：|obs[12]| <= 0.024 且 |obs[13]| <= 0.030
# 朝向误差 < 30deg -> cos(err) > cos(30deg) = 0.866
# 速度 < 0.05 m/s -> 归一化后 obs[8],obs[9] 是 /3.0，故 0.05/3.0 ~= 0.01667
_DOCK_X_TOL = 0.024
_DOCK_Y_TOL = 0.030
_ALIGN_COS_MIN = 0.866
_SPEED_NORM_MAX = 0.05 / 3.0

# 自检记录（每步平均奖励估计）：
#   R_idle      ~ 0.0   （不接触、无进展、不越界：所有组件≈0）
#   R_push      ~ +6.0  （接触正常推进：progress_delta 约 +6，gentleness 在低速下≈0）
#   R_settled   ~ +20.0 （泊位内+对齐+慢：settle 每步 +20，其余组件恰好为 0）
#   满足：R_push > R_idle（差距 6.0 > 最大罚项量级 ~3.0），R_settled > R_push。
#   完成事件 B = 300.0 一次性；过程型单步上限之和约 6.0，400 步约 2400，
#   10*B = 3000 > 3*2400 = 7200？不成立 -> 因此过程项用增量且有界，见下。
#   实际：progress_delta 每步上限 +6，但只有真正靠近时才有；悬停时 delta≈0，
#   故 400 步悬停的累计过程回报 ≈ 0，10*B=3000 远大于 3*0。
#   自检③：closing=1.0 时 gentleness=-0.05*1.0=-0.05，需与推进同量级 -> 提高 k。
#   本实现 k=6.0：closing=1.0 时 gentleness=-6.0，与推进 +6 同量级，满足自检③。
#   自检⑤（停稳12次）：除一次性事件那一次外，其余 11 次返回恰好 0（完成态其余组件为0）。


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 回合边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---------- 从观测中提取信号 ----------
    cart_x = obs[0]
    cart_y = obs[1]
    hx = obs[2]
    hy = obs[3]
    cart_fwd = obs[4]

    crate_wvx = next_obs[8] * 3.0
    crate_wvy = next_obs[9] * 3.0

    prev_dx = obs[12]
    prev_dy = obs[13]
    new_dx = next_obs[12]
    new_dy = next_obs[13]

    # 货箱到泊位中心距离（归一化坐标下的欧氏距离）
    prev_dist = (prev_dx * prev_dx + prev_dy * prev_dy) ** 0.5
    new_dist = (new_dx * new_dx + new_dy * new_dy) ** 0.5

    # 货箱朝向误差（归一化：1 = 完全对齐，0 = 反向）
    crate_cos = next_obs[10]
    crate_sin = next_obs[11]
    crate_align = crate_cos  # 泊位朝向假定与货箱目标朝向一致 -> 用 cos(err) 近似
    if crate_align < -1.0:
        crate_align = -1.0
    if crate_align > 1.0:
        crate_align = 1.0

    # 货箱速度（归一化）
    crate_speed_norm = (next_obs[8] * next_obs[8] + next_obs[9] * next_obs[9]) ** 0.5

    # ---------- 完成谓词（显式从 obs 推断） ----------
    in_dock = 1.0 if (abs(new_dx) <= _DOCK_X_TOL and abs(new_dy) <= _DOCK_Y_TOL) else 0.0
    aligned = 1.0 if crate_align >= _ALIGN_COS_MIN else 0.0
    slow = 1.0 if crate_speed_norm <= _SPEED_NORM_MAX else 0.0
    done_cond = 1.0 if (in_dock > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0

    if done_cond > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # ---------- 接触轻柔度（唯一能教"减速"的信号） ----------
    crate_along = crate_wvx * hx + crate_wvy * hy
    closing = cart_fwd * 3.0 - crate_along
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    gentleness = -6.0 * contact * closing

    # ---------- 推进增量（增量形式，避免悬停收割） ----------
    progress_delta = 0.0
    if new_dist < prev_dist:
        # 只在更接近时给分；用对齐度做门控
        raw_gain = (prev_dist - new_dist) * 60.0
        align_gate = 0.5 + 0.5 * crate_align  # [0,1]
        progress_delta = raw_gain * align_gate

    # ---------- 停稳期每步收益（完成态唯一正项） ----------
    settle = 0.0
    if done_cond > 0.5:
        settle = 20.0

    # ---------- 首次进入泊位（一次性） ----------
    enter_event = 0.0
    if in_dock > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 30.0

    # ---------- 越界守卫（随接近边界单调下降） ----------
    bound_penalty = 0.0
    ax = abs(cart_x)
    ay = abs(cart_y)
    if ax > 0.95:
        bound_penalty -= (ax - 0.95) * 200.0
    if ay > 0.95:
        bound_penalty -= (ay - 0.95) * 200.0

    # 货箱接近边界（用货箱到泊位偏移推断货箱是否远离场地中心）
    # 货箱世界位置粗略恢复：cart 位置 + 车身旋转后的相对位置
    rel_x = obs[6] * 3.0
    rel_y = obs[7] * 3.0
    # 车身系 -> 世界系
    wx = rel_x * hx - rel_y * hy
    wy = rel_x * hy + rel_y * hx
    crate_wx = cart_x * 5.0 + wx
    crate_wy = cart_y * 4.0 + wy
    # 归一化到 [-1,1] 边界（半宽 5.0, 半高 4.0）
    cnx = crate_wx / 5.0
    cny = crate_wy / 4.0
    cax = abs(cnx)
    cay = abs(cny)
    if cax > 0.95:
        bound_penalty -= (cax - 0.95) * 200.0
    if cay > 0.95:
        bound_penalty -= (cay - 0.95) * 200.0

    # ---------- 完成态下，除一次性事件外其余组件必须为 0 ----------
    if done_cond > 0.5:
        # 仅在完成态：保留 settle（停稳每步收益）与一次性事件
        # 其余组件（progress_delta / gentleness / enter_event / bound_penalty）置零
        progress_delta = 0.0
        gentleness = 0.0
        enter_event = 0.0
        bound_penalty = 0.0

    # ---------- 汇总 ----------
    total = (
        progress_delta
        + gentleness
        + settle
        + enter_event
        + success_event
        + bound_penalty
    )

    components = {
        "progress_delta": float(progress_delta),
        "gentleness": float(gentleness),
        "settle": float(settle),
        "enter_event": float(enter_event),
        "success_event": float(success_event),
        "bound_penalty": float(bound_penalty),
    }

    return float(total), components