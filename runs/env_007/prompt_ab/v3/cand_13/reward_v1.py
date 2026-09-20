def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 回合边界检测（obs[18] 单调递增，重置时回落）----
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---- 泊位几何（由环境事实给出的缩放恢复）----
    # obs[12]: 货箱中心到泊位中心有符号 x 偏移 / 仓库半宽
    # obs[13]: 货箱中心到泊位中心有符号 y 偏移 / 仓库半高
    dx = float(next_obs[12]) * 5.0
    dy = float(next_obs[13]) * 4.0
    dist = (dx * dx + dy * dy) ** 0.5

    pdx = float(obs[12]) * 5.0
    pdy = float(obs[13]) * 4.0
    prev_dist = (pdx * pdx + pdy * pdy) ** 0.5

    # ---- 主信号 1：货箱向泊位的增量进展（只在更接近时给分）----
    progress = prev_dist - dist
    if progress > 0.0:
        crate_progress = 6.0 * progress
    else:
        crate_progress = 0.0

    # ---- 主信号 2：接近泊位时的朝向对齐（门控，仅近距离激活）----
    # 货箱朝向误差
    crate_ang = 0.0
    c_cos = float(next_obs[10])
    c_sin = float(next_obs[11])
    crate_ang = (c_sin * c_sin) ** 0.5  # |sin(theta)|，theta 相对 0 轴
    # 用 |sin| 作为朝向偏差代理（对齐时为 0）
    align_err = crate_ang
    if align_err > 1.0:
        align_err = 1.0
    # 门控：仅在货箱接近泊位时启用
    near_gate = 1.0 / (1.0 + 1.0 * dist)
    align_term = 0.8 * near_gate * (1.0 - align_err)

    # ---- 主信号 3：货箱在泊位附近时的低速（门控，仅近距离激活）----
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    slow_factor = 1.0 / (1.0 + 2.0 * crate_speed)
    slow_term = 0.8 * near_gate * slow_factor

    # ---- 接触轻柔度（核心技能信号）----
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---- 越界 hinge 惩罚（小车与货箱）----
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    oob = 0.0
    if cart_x > 0.95:
        oob += (cart_x - 0.95)
    if cart_x < -0.95:
        oob += (-0.95 - cart_x)
    if cart_y > 0.95:
        oob += (cart_y - 0.95)
    if cart_y < -0.95:
        oob += (-0.95 - cart_y)
    out_of_bounds = -2.0 * oob

    # ---- 障碍接近惩罚（hinge，仅在很近时）----
    obs_front = float(next_obs[15])
    obs_left = float(next_obs[16])
    obs_right = float(next_obs[17])
    obs_pen = 0.0
    if obs_front > 0.85:
        obs_pen += (obs_front - 0.85)
    if obs_left > 0.85:
        obs_pen += (obs_left - 0.85)
    if obs_right > 0.85:
        obs_pen += (obs_right - 0.85)
    obstacle_penalty = -1.0 * obs_pen

    # ---- 完成条件判定（严格使用环境事实给出的容差）----
    inside_x = 1.0 if abs(float(next_obs[12])) <= 0.024 else 0.0
    inside_y = 1.0 if abs(float(next_obs[13])) <= 0.030 else 0.0
    # 朝向误差 < 30°：|sin(theta)| < sin(30°) = 0.5
    aligned = 1.0 if crate_ang < 0.5 else 0.0
    slow_enough = 1.0 if crate_speed < 0.05 else 0.0

    cond_ok = (inside_x > 0.5 and inside_y > 0.5 and aligned > 0.5 and slow_enough > 0.5)

    if cond_ok:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---- 一次性首次进入泊位奖励 ----
    enter_bonus = 0.0
    if inside_x > 0.5 and inside_y > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_bonus = 40.0

    # ---- 一次性完成事件奖励 ----
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    components = {}
    components["crate_progress"] = crate_progress
    components["align_term"] = align_term
    components["slow_term"] = slow_term
    components["gentleness"] = gentleness
    components["out_of_bounds"] = out_of_bounds
    components["obstacle_penalty"] = obstacle_penalty
    components["enter_bonus"] = enter_bonus
    components["success_event"] = success_event

    total_reward = (
        crate_progress
        + align_term
        + slow_term
        + gentleness
        + out_of_bounds
        + obstacle_penalty
        + enter_bonus
        + success_event
    )
    return (float(total_reward), components)


_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]