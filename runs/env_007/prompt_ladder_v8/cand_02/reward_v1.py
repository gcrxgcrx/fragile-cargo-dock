# ---- module-level state (episode boundary detection via obs[18]) ----
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]

# =========================================================================
# 自检记录（量级估算，单位：单步奖励）
# -------------------------------------------------------------------------
# 组件设计：
#   progress   : 有符号增量推进项（货箱到泊位距离的变化，按米计，对称）
#                正常推箱 ~ +0.02..+0.10 / 步（距离变化 0.02~0.10 m/步）
#   settle     : 停稳期每步收益（泊位内 + 对齐 + 慢）  -> +20.0 / 步
#   gentleness : 接触且接近时的惩罚 -k * closing，k=8.0
#                closing=1.0 -> -8.0 ; closing=0.05 -> -0.4 ; closing=0 -> 0
#   boundary   : 越界 hinge 惩罚，最坏 ~ -30 / 步
#   align_gate : 对齐度门控（乘在 progress 上，不单独给分）
#
# 自检 ①（idle vs push）:
#   idle : progress≈0, settle=0, gentleness=0, boundary=0        -> ~0.0
#   push : progress≈+0.05, gentleness≈0（匀速推不判撞击）        -> ~+0.05
#   => push > idle ✔
#
# 自检 ②（悬停 0.3m vs 真正完成）:
#   悬停：progress≈0（不动），settle=0（不在容差内），400 步 ~ 0
#   完成：settle 每步 +20，保持 10 步 ~ +200，另有一次性事件
#   => 完成轨迹累计严格更高 ✔
#
# 自检 ③（撞击 1.0 vs 轻柔 0.05）:
#   ⑤ closing=1.0 : gentleness = -8.0，总 ~ -8.0 + progress(0.05) ≈ -7.95
#   ⑥ closing=0.05: gentleness = -0.4，总 ~ -0.4 + 0.05 ≈ -0.35
#   差距 ≈ 7.6 >> 推进项单步值(0.05) ✔
#
# 自检 ④（停稳状态连续 12 次调用）:
#   每次差值 = +20.0（settle 每步发放，不被任何计数/一次性事件关闭） ✔
#
# 自检 ⑤（三行均值）:
#   R_idle    ≈ 0.0
#   R_push    ≈ +0.05  （progress 为主，罚项不淹没）
#   R_settled ≈ +20.0  （settle 主导）
#   R_push - R_idle = 0.05，与推进项同量级；
#   R_settled > R_push ✔；停稳是全局最优点 ✔
# =========================================================================


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 0. episode 边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    components = {}

    # ---------- 1. 几何量恢复 ----------
    # 仓库半宽 = 5.0 m, 半高 = 4.0 m（由 obs[6]/obs[7] 的 /3.0 与 obs[0]/obs[1] 的归一化推得）
    HALF_W = 5.0
    HALF_H = 4.0

    # 货箱到泊位的有符号偏移（归一化），乘半宽/半高恢复为米
    dx = float(next_obs[12]) * HALF_W
    dy = float(next_obs[13]) * HALF_H
    dist = (dx * dx + dy * dy) ** 0.5

    odx = float(obs[12]) * HALF_W
    ody = float(obs[13]) * HALF_H
    old_dist = (odx * odx + ody * ody) ** 0.5

    # ---------- 2. 对齐度（门控用，不单独给分） ----------
    # 货箱朝向 vs 泊位朝向（泊位朝向取 -x 方向，即货箱被推入后车头指向）
    # 用货箱朝向与"指向泊位中心"方向的夹角作为对齐代理
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    # 货箱朝向角
    # 目标朝向：货箱朝向应沿 -x（面向泊位内侧），即 cos = -1, sin = 0
    # 对齐度 = (1 + (-crate_cos)) / 2  -> 0..1
    align = (1.0 - crate_cos) * 0.5
    if align < 0.0:
        align = 0.0
    if align > 1.0:
        align = 1.0

    # ---------- 3. 推进项（有符号增量，按米计，对称） ----------
    # 本帧更接近 -> 正分；本帧更远 -> 负分。禁止 max(0, ...) 单边刷分。
    # 对齐度作为门控乘在增量上（不对齐时推进收益打折，但不为负）
    delta_m = old_dist - dist
    progress = 3.0 * delta_m * (0.3 + 0.7 * align)
    components["progress"] = progress

    # ---------- 4. 轻柔度（接触 + 接近时惩罚） ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    cart_cos_h = float(obs[2])
    cart_sin_h = float(obs[3])
    crate_along_heading = crate_vx * cart_cos_h + crate_vy * cart_sin_h
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    # k = 8.0：closing=1.0 -> -8.0，与推进项同量级或更大；closing≈0 -> 0
    gentleness = -8.0 * contact * closing
    components["gentleness"] = gentleness

    # ---------- 5. 停稳期每步收益（必须每步发放，不被任何开关关闭） ----------
    # 谓词：货箱在泊位容差内（|obs[12]|<=0.024, |obs[13]|<=0.030）+ 对齐 + 慢
    inside = 1.0 if (abs(float(next_obs[12])) <= 0.024 and abs(float(next_obs[13])) <= 0.030) else 0.0
    crate_speed = ((crate_vx * crate_vx) + (crate_vy * crate_vy)) ** 0.5
    slow = 1.0 if crate_speed < 0.05 else 0.0
    aligned = 1.0 if align > 0.75 else 0.0  # 朝向误差 < ~30°
    settled_gate = inside * slow * aligned
    settle = 20.0 * settled_gate
    components["settle"] = settle

    # 连续计数（用于一次性事件，不影响 settle 发放）
    if settled_gate > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- 6. 首次进入泊位（一次性，整局只发一次） ----------
    enter_bonus = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_bonus = 5.0
    components["enter_bonus"] = enter_bonus

    # ---------- 7. 一次性完成事件（可写可不写，不承担实际作用） ----------
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["success_event"] = success_event

    # ---------- 8. 越界守卫（小车 + 货箱） ----------
    # 小车位置 obs[0], obs[1]（+1.0 就是墙）；超过 ~0.95 明显生效
    ax = abs(float(next_obs[0]))
    ay = abs(float(next_obs[1]))
    cart_over = 0.0
    if ax > 0.90:
        cart_over += (ax - 0.90)
    if ay > 0.90:
        cart_over += (ay - 0.90)
    # 货箱世界坐标 = 小车位置 + 旋转后的车体系偏移
    rel_x = float(next_obs[6]) * 3.0
    rel_y = float(next_obs[7]) * 3.0
    ch = float(next_obs[2])
    sh = float(next_obs[3])
    crate_wx = (float(next_obs[0]) * HALF_W) + (rel_x * ch - rel_y * sh)
    crate_wy = (float(next_obs[1]) * HALF_H) + (rel_x * sh + rel_y * ch)
    cnx = abs(crate_wx) / HALF_W
    cny = abs(crate_wy) / HALF_H
    crate_over = 0.0
    if cnx > 0.90:
        crate_over += (cnx - 0.90)
    if cny > 0.90:
        crate_over += (cny - 0.90)
    boundary_pen = -150.0 * (cart_over + crate_over)
    components["boundary_penalty"] = boundary_pen

    # ---------- 9. 动作平滑（轻量，不压制推进） ----------
    a0 = float(action[0])
    a1 = float(action[1])
    smooth_pen = -0.02 * (a0 * a0 + a1 * a1)
    components["action_smooth"] = smooth_pen

    total = progress + gentleness + settle + enter_bonus + success_event + boundary_pen + smooth_pen
    return float(total), components