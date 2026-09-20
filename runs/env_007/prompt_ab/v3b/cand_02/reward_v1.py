_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode 边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---------- 泊位几何量 ----------
    # 泊位容差（环境事实）：|dx| <= 0.024, |dy| <= 0.030
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    pdx = float(obs[12])
    pdy = float(obs[13])
    prev_dist = (pdx * pdx + pdy * pdy) ** 0.5

    # ---------- 货箱速度 ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    # ---------- 朝向误差 ----------
    # 货箱朝向 vs 泊位朝向（泊位朝向按 0 处理，即 cos=1, sin=0）
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    # 朝向误差角（rad），用 |atan2(sin,cos)| 近似
    ang_err = abs(crate_sin)  # sin 在小角度下近似角度
    align = 1.0 - min(1.0, ang_err / 0.5236)  # 30° = 0.5236 rad

    # ---------- 主信号 1：货箱到泊位推进量（增量形式） ----------
    progress = prev_dist - dist
    if progress > 0.05:
        progress = 0.05
    if progress < -0.05:
        progress = -0.05
    r_progress = 8.0 * progress

    # ---------- 主信号 2：接近泊位时的对齐 shaping（门控，仅接近时激活） ----------
    near_gate = 1.0 / (1.0 + 8.0 * dist)  # dist 小 → 接近 1
    r_align = 1.0 * near_gate * align

    # ---------- 主信号 3：接近泊位时的低速 shaping（门控，仅接近时激活） ----------
    # 目标速度：接近泊位时希望货箱慢
    speed_target = 0.05
    speed_excess = crate_speed - speed_target
    if speed_excess < 0.0:
        speed_excess = 0.0
    r_slow = -1.5 * near_gate * (speed_excess ** 2)

    # ---------- 接触轻柔度（核心技能信号） ----------
    cart_vx_w = float(next_obs[2]) * float(next_obs[4]) * 3.0
    cart_vy_w = float(next_obs[3]) * float(next_obs[4]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---------- 越界 hinge 惩罚 ----------
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    # 车在仓库内，|x|<1, |y|<1；接近边界时惩罚
    oob = 0.0
    if abs(cart_x) > 0.9:
        oob += (abs(cart_x) - 0.9) ** 2
    if abs(cart_y) > 0.9:
        oob += (abs(cart_y) - 0.9) ** 2
    r_oob = -5.0 * oob

    # ---------- 动作平滑（轻量） ----------
    r_smooth = -0.02 * (float(action[0]) ** 2 + float(action[1]) ** 2)

    # ---------- 完成事件（一次性） ----------
    # 完成条件：完全进入泊位 + 朝向 < 30° + 速度 < 0.05
    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0
    aligned = 1.0 if ang_err < 0.5236 else 0.0
    slow = 1.0 if crate_speed < 0.05 else 0.0

    if inside > 0.5 and aligned > 0.5 and slow > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # 首次进入泊位的一次性奖励
    enter_event = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 20.0

    # ---------- 汇总 ----------
    components = {
        "progress": r_progress,
        "align": r_align,
        "slow": r_slow,
        "gentleness": gentleness,
        "oob": r_oob,
        "smooth": r_smooth,
        "enter_event": enter_event,
        "success_event": success_event,
    }

    total = (
        r_progress
        + r_align
        + r_slow
        + gentleness
        + r_oob
        + r_smooth
        + enter_event
        + success_event
    )

    return (float(total), components)