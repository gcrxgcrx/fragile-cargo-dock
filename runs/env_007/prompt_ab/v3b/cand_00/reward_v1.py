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

    # ---------- 基础量 ----------
    # 货箱到泊位的有符号偏移（归一化）
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    # 归一化距离（无量纲）
    dist = (dx * dx + dy * dy) ** 0.5

    # 上一帧距离
    pdx = float(obs[12])
    pdy = float(obs[13])
    prev_dist = (pdx * pdx + pdy * pdy) ** 0.5

    # 货箱速度（世界系，归一化）
    cvx = float(next_obs[8])
    cvy = float(next_obs[9])
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向误差（弧度）
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    # 泊位朝向假定为 +x 轴方向（cos=1, sin=0）
    heading_err = (sh * sh) ** 0.5  # |sin(theta_crate)| 作为朝向误差代理
    # 更精确：用 atan2 近似，|sin| 在 [0,1]，30° 对应 sin(30°)=0.5
    align_ok = 1.0 - min(1.0, heading_err / 0.5)  # 0 表示完全不对齐，1 表示对齐

    # 接触标志
    contact = 1.0 if next_obs[14] > 0.5 else 0.0

    # ---------- 组件 1：货箱向泊位推进（增量形式） ----------
    delta = prev_dist - dist  # 正 = 更接近
    progress = 2.0 * delta  # 主推进信号

    # ---------- 组件 2：轻柔接触（接近速度惩罚） ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    gentleness = -0.05 * contact * closing

    # ---------- 组件 3：接近泊位时的速度抑制（门控，仅在接近时激活） ----------
    # 仅在泊位附近（dist < 0.15）激活，且只在货箱有速度时惩罚
    near_dock_gate = 1.0 if dist < 0.15 else 0.0
    speed_penalty = -0.5 * near_dock_gate * crate_speed

    # ---------- 组件 4：朝向对齐 shaping（仅在泊位内激活） ----------
    inside_gate = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0
    align_bonus = 0.3 * inside_gate * align_ok

    # ---------- 组件 5：边界惩罚（hinge，只在越界附近） ----------
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    edge_pen = 0.0
    if abs(cart_x) > 0.9:
        edge_pen -= 0.5 * (abs(cart_x) - 0.9)
    if abs(cart_y) > 0.9:
        edge_pen -= 0.5 * (abs(cart_y) - 0.9)

    # ---------- 组件 6：动作平滑（轻量） ----------
    smooth = -0.01 * (float(action[0]) ** 2 + float(action[1]) ** 2)

    # ---------- 完成事件检测 ----------
    # 完全进入泊位：|dx| <= 0.024, |dy| <= 0.030
    # 朝向误差 < 30°：|sin| < 0.5
    # 速度 < 0.05 m/s：crate_speed < 0.05/3.0
    in_dock = (abs(dx) <= 0.024) and (abs(dy) <= 0.030)
    aligned = heading_err < 0.5
    slow = crate_speed < (0.05 / 3.0)

    if in_dock and aligned and slow:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # 首次进入泊位（一次性）
    entered_bonus = 0.0
    if in_dock and not _ENTERED[0]:
        _ENTERED[0] = True
        entered_bonus = 20.0

    # 完成事件（一次性）
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # ---------- 汇总 ----------
    components = {
        "progress": progress,
        "gentleness": gentleness,
        "speed_penalty": speed_penalty,
        "align_bonus": align_bonus,
        "edge_penalty": edge_pen,
        "smoothness": smooth,
        "entered_bonus": entered_bonus,
        "success_event": success_event,
    }

    total = (
        progress
        + gentleness
        + speed_penalty
        + align_bonus
        + edge_pen
        + smooth
        + entered_bonus
        + success_event
    )

    return (float(total), components)