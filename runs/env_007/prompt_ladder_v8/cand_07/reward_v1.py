# 模块级状态：用于回合边界检测与一次性事件发放
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_DIST = [-1.0]

# 自检记录（量级估计，单位：单步奖励）
# 正常推箱（接触、closing≈0.05、货箱向泊位推进 0.01m/步）:
#   progress ≈ 0.01 * 100 = 1.0, settle = 0, gentleness ≈ -0.05*0.05 ≈ -0.0025
#   R_push ≈ 1.0
# 什么都不做（无接触、货箱静止、距离不变）:
#   progress = 0, settle = 0, gentleness = 0, 边界/障碍罚 ≈ 0
#   R_idle ≈ 0.0
# 停稳在泊位内（谓词成立）:
#   settle = +20, progress 增量 ≈ 0, gentleness ≈ 0
#   R_settled ≈ 20.0
# 满足: R_push(1.0) > R_idle(0.0)，差 1.0 ≥ 最大罚项量级(~0.5)；
#       R_settled(20) > R_push(1.0)。


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 回合边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = -1.0
    _PREV_T[0] = t

    components = {}

    # ---------- 几何量（恢复到米制） ----------
    HALF_W = 5.0   # 仓库半宽（米），用于 obs[12] -> 米
    HALF_H = 4.0   # 仓库半高（米），用于 obs[13] -> 米

    dx_m = float(next_obs[12]) * HALF_W   # 货箱到泊位 x 偏移（米）
    dy_m = float(next_obs[13]) * HALF_H   # 货箱到泊位 y 偏移（米）
    dist = (dx_m * dx_m + dy_m * dy_m) ** 0.5

    prev_dx_m = float(obs[12]) * HALF_W
    prev_dy_m = float(obs[13]) * HALF_H
    prev_dist = (prev_dx_m * prev_dx_m + prev_dy_m * prev_dy_m) ** 0.5

    # ---------- 1) 有符号推进项（增量、按米计） ----------
    # 本帧更接近给正分，更远给负分（对称）。禁止只奖励接近不惩罚远离。
    progress = 100.0 * (prev_dist - dist)
    components["progress"] = progress

    # ---------- 2) 轻柔度：接触中且接近时的惩罚 ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    cart_cos = float(obs[2])
    cart_sin = float(obs[3])
    crate_along_heading = crate_vx * cart_cos + crate_vy * cart_sin
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    # k=1.0：closing=1.0 m/s 时惩罚 -1.0，与推进项同量级
    gentleness = -1.0 * contact * closing
    components["gentleness"] = gentleness

    # ---------- 3) 停稳期每步收益（谓词成立即每步发放，不被任何开关关闭） ----------
    # 谓词：货箱完全在泊位容差内 + 朝向对齐 < 30° + 速度 < 0.05 m/s
    in_dock = (abs(float(next_obs[12])) <= 0.024) and (abs(float(next_obs[13])) <= 0.030)

    crate_theta = 0.0
    c_cos = float(next_obs[10])
    c_sin = float(next_obs[11])
    # 货箱朝向误差：以 atan2(sin, cos) 的绝对值近似（泊位朝向按 0 处理，取相对量级）
    crate_theta = c_sin if c_sin >= 0.0 else -c_sin
    aligned = crate_theta <= 0.5   # sin(30°) ≈ 0.5

    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    slow = crate_speed < 0.05

    settle_ok = in_dock and aligned and slow
    settle = 20.0 if settle_ok else 0.0
    components["settle"] = settle

    # 连续计数（用于一次性事件，可选保留）
    if settle_ok:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- 4) 首次进入泊位（一次性事件，整局只发一次） ----------
    enter_bonus = 0.0
    if in_dock and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_bonus = 20.0
    components["enter_bonus"] = enter_bonus

    # ---------- 5) 完成事件（可选，一次性；本环境单步裁剪使其不承担主要作用） ----------
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["success_event"] = success_event

    # ---------- 6) 越界守卫（小车与货箱） ----------
    cart_x = abs(float(next_obs[0]))
    cart_y = abs(float(next_obs[1]))

    oob = 0.0
    # 小车越界：|x| 或 |y| > 0.95 起明显生效
    cart_margin = cart_x if cart_x > cart_y else cart_y
    if cart_margin > 0.95:
        oob -= 50.0 * (cart_margin - 0.95)
    components["cart_oob"] = oob

    # 货箱越界：由 obs[6],obs[7] 恢复货箱世界坐标
    rel_x = float(next_obs[6]) * 3.0
    rel_y = float(next_obs[7]) * 3.0
    crate_wx = float(next_obs[0]) * HALF_W + rel_x * cart_cos - rel_y * cart_sin
    crate_wy = float(next_obs[1]) * HALF_H + rel_x * cart_sin + rel_y * cart_cos
    crate_margin = (crate_wx / HALF_W) if (crate_wx / HALF_W) >= 0.0 else -(crate_wx / HALF_W)
    tmp = (crate_wy / HALF_H) if (crate_wy / HALF_H) >= 0.0 else -(crate_wy / HALF_H)
    if tmp > crate_margin:
        crate_margin = tmp
    crate_oob = 0.0
    if crate_margin > 0.95:
        crate_oob -= 50.0 * (crate_margin - 0.95)
    components["crate_oob"] = crate_oob

    # ---------- 7) 障碍接近惩罚（hinge，仅在很近时生效） ----------
    obs_pen = 0.0
    sf = float(next_obs[15])
    sl = float(next_obs[16])
    sr = float(next_obs[17])
    # 仅在接触/极近且未推箱时轻罚，避免抑制正常推箱
    if contact < 0.5:
        if sf > 0.9:
            obs_pen -= 0.5 * (sf - 0.9)
        if sl > 0.9:
            obs_pen -= 0.5 * (sl - 0.9)
        if sr > 0.9:
            obs_pen -= 0.5 * (sr - 0.9)
    components["obstacle"] = obs_pen

    # ---------- 8) 动作平滑（轻量，不压制推动） ----------
    smooth = -0.05 * (float(action[1]) * float(action[1]))
    components["smooth"] = smooth

    total = (
        progress
        + gentleness
        + settle
        + enter_bonus
        + success_event
        + oob
        + crate_oob
        + obs_pen
        + smooth
    )

    return (float(total), components)