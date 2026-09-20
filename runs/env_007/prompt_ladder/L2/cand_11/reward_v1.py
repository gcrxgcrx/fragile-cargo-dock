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

    # ---------- 几何量（从 obs 恢复，单位：归一化） ----------
    # 货箱到泊位的有符号偏移（归一化）
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    odx = float(obs[12])
    ody = float(obs[13])

    # 到泊位中心的归一化距离
    dist = (dx * dx + dy * dy) ** 0.5
    odist = (odx * odx + ody * ody) ** 0.5

    # 完成容差（环境事实给出，不得放宽）
    in_dock = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0

    # 朝向误差：货箱朝向 vs 泊位朝向（假设泊位朝向为 +x，即 cos=1, sin=0）
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    # 朝向误差余弦（1 表示对齐）
    align_cos = crate_cos
    if align_cos < 0.0:
        align_cos = 0.0
    # 30° 阈值对应 cos ≈ 0.866
    aligned = 1.0 if align_cos >= 0.866 else 0.0

    # 货箱速度（世界系，m/s）
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    slow = 1.0 if crate_speed < 0.05 else 0.0

    # ---------- 完成条件（显式推断） ----------
    done_now = 1.0 if (in_dock > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0
    if done_now > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- 组件字典 ----------
    components = {}

    # ===== 1) 推进增量（主过程信号，增量形式，避免悬停收割） =====
    progress = odist - dist  # 本帧靠近泊位的归一化距离减少量
    # 归一化尺度：整场约 1.0，单步典型 0.001~0.01，放大到与惩罚同量级
    progress_reward = 6.0 * progress
    if progress_reward < -0.5:
        progress_reward = -0.5
    if progress_reward > 0.5:
        progress_reward = 0.5
    components["crate_to_dock_progress"] = progress_reward

    # ===== 2) 轻柔接触惩罚（唯一能教减速的信号） =====
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    # k = 1.2：closing=1.0 m/s 时惩罚 -1.2，与推进项同量级或更大
    gentleness = -1.2 * contact * closing
    components["soft_contact"] = gentleness

    # ===== 3) 接近泊位时的速度抑制（门控在接近时才生效，避免阻碍到达） =====
    # 只在货箱已接近泊位（归一化距离 < 0.15）时激活
    near_gate = 0.0
    if dist < 0.15:
        near_gate = (0.15 - dist) / 0.15
        if near_gate < 0.0:
            near_gate = 0.0
        if near_gate > 1.0:
            near_gate = 1.0
    # 速度超阈值时惩罚（hinge 形式，慢速不罚）
    speed_excess = crate_speed - 0.05
    if speed_excess < 0.0:
        speed_excess = 0.0
    speed_penalty = -0.8 * near_gate * speed_excess
    components["crate_speed_near_dock"] = speed_penalty

    # ===== 4) 越界守卫（小车 + 货箱，随接近边界单调下降） =====
    # 小车位置
    cx = float(next_obs[0])
    cy = float(next_obs[1])
    acx = cx if cx >= 0.0 else -cx
    acy = cy if cy >= 0.0 else -cy
    cart_margin = 0.0
    if acx > 0.85:
        cart_margin = acx - 0.85
    if acy > 0.85:
        m = acy - 0.85
        if m > cart_margin:
            cart_margin = m
    # 货箱世界位置（近似）：由 obs[6],obs[7] 车体系 + 小车位置恢复
    # 简化：用货箱到泊位偏移 + 泊位位置不可知，改用货箱相对小车 + 小车位置
    rel_x = float(next_obs[6]) * 3.0
    rel_y = float(next_obs[7]) * 3.0
    ch = float(next_obs[2])
    sh = float(next_obs[3])
    crate_wx = cx * 5.0 + rel_x * ch - rel_y * sh
    crate_wy = cy * 4.0 + rel_x * sh + rel_y * ch
    # 归一化到半宽/半高
    crate_nx = crate_wx / 5.0
    crate_ny = crate_wy / 4.0
    acnx = crate_nx if crate_nx >= 0.0 else -crate_nx
    acny = crate_ny if crate_ny >= 0.0 else -crate_ny
    crate_margin = 0.0
    if acnx > 0.90:
        crate_margin = acnx - 0.90
    if acny > 0.90:
        m = acny - 0.90
        if m > crate_margin:
            crate_margin = m
    worst_margin = cart_margin if cart_margin > crate_margin else crate_margin
    oob_penalty = -40.0 * worst_margin
    components["out_of_bounds"] = oob_penalty

    # ===== 5) 完成状态下的清零：除一次性事件外其余组件必须为 0 =====
    if done_now > 0.5:
        components["crate_to_dock_progress"] = 0.0
        components["soft_contact"] = 0.0
        components["crate_speed_near_dock"] = 0.0
        components["out_of_bounds"] = 0.0

    # ===== 6) 首次进入泊位（一次性） =====
    entered_bonus = 0.0
    if in_dock > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        entered_bonus = 30.0
    components["first_enter_dock"] = entered_bonus

    # ===== 7) 完成事件（一次性，连续 10 步） =====
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["docked_success"] = success_event

    total = 0.0
    for k in components:
        total += components[k]

    return (float(total), components)