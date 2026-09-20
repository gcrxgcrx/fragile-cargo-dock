_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 回合边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---------- 泊位几何 ----------
    # obs[12]: 货箱到泊位中心有符号 x 偏移 / 半宽 ; obs[13]: / 半高
    # 容差: |obs[12]| <= 0.024 且 |obs[13]| <= 0.030 (完全进入)
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    pdx = float(obs[12])
    pdy = float(obs[13])
    prev_dist = (pdx * pdx + pdy * pdy) ** 0.5

    # ---------- 货箱朝向误差 ----------
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    # 朝向误差角(rad) 相对泊位朝向(假设为 0 -> cos=1)
    # 用 atan2 近似: 误差 ~ |sin| 主导
    heading_err = abs(crate_sin)
    heading_ok = heading_err < 0.5  # sin(30deg)=0.5

    # ---------- 货箱速度 ----------
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    speed_ok = crate_speed < 0.05

    # ---------- 完成条件 ----------
    inside = (abs(dx) <= 0.024) and (abs(dy) <= 0.030)
    done_cond = inside and heading_ok and speed_ok
    if done_cond:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    components = {}

    # ---------- 1. 主推进信号: 增量式接近 (只在更接近时给分) ----------
    progress = prev_dist - dist  # >0 表示这一帧更接近泊位
    if progress < 0.0:
        progress = 0.0
    components["crate_to_dock_progress"] = 8.0 * progress

    # ---------- 2. 朝向对齐 shaping (仅在接近泊位时激活, 作为门控乘子) ----------
    # 用接近度做门控: 离泊位越近越关注朝向
    near_gate = 1.0 / (1.0 + 20.0 * dist)
    align_bonus = 0.5 * near_gate * (1.0 - heading_err)
    if align_bonus < 0.0:
        align_bonus = 0.0
    components["crate_docking_quality"] = align_bonus

    # ---------- 3. 接近泊位时抑制货箱速度 (仅在泊位附近启用) ----------
    # 仅在 dist < 0.15 时启用, 避免阻碍到达
    if dist < 0.15:
        speed_pen = -1.0 * near_gate * crate_speed
    else:
        speed_pen = 0.0
    components["crate_speed_penalty_near_dock"] = speed_pen

    # ---------- 4. 轻柔接触信号 ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing
    components["soft_contact"] = gentleness

    # ---------- 5. 越界惩罚 (hinge, 小车接近边界) ----------
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    oob = 0.0
    if abs(cart_x) > 0.9:
        oob -= 1.0 * (abs(cart_x) - 0.9)
    if abs(cart_y) > 0.9:
        oob -= 1.0 * (abs(cart_y) - 0.9)
    components["out_of_bounds"] = oob

    # ---------- 6. 首次进入泊位一次性奖励 ----------
    enter_bonus = 0.0
    if inside and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_bonus = 20.0
    components["first_enter_dock"] = enter_bonus

    # ---------- 7. 完成事件 (一次性, 大额) ----------
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["success_event"] = success_event

    total = 0.0
    for k in components:
        total += components[k]

    return (float(total), components)