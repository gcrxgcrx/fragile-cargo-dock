```python
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

    # ---------- 几何量（泊位偏移，归一化坐标） ----------
    # 容差：|obs[12]| <= 0.024, |obs[13]| <= 0.030
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dx_prev = float(obs[12])
    dy_prev = float(obs[13])

    # 归一化距离（用容差尺度归一，便于门控）
    ex = dx / 0.024
    ey = dy / 0.030
    dist_n = (ex * ex + ey * ey) ** 0.5
    ex_p = dx_prev / 0.024
    ey_p = dy_prev / 0.030
    dist_n_prev = (ex_p * ex_p + ey_p * ey_p) ** 0.5

    # ---------- 货箱朝向对齐 ----------
    # 货箱朝向 cos/sin；泊位对齐方向未显式给出，用货箱朝向与"世界轴"的对齐作为代理：
    # 用 |cos| 衡量轴向对齐（方形货箱 90° 对称），误差 < 30° 即 |cos| > cos(30°)=0.866
    ccos = float(next_obs[10])
    csin = float(next_obs[11])
    align = (ccos * ccos - csin * csin)  # cos(2*theta)，方形货箱周期为 90°
    if align < 0.0:
        align = 0.0
    # 对齐门：误差<30° 时接近 1
    align_gate = align
    if align_gate > 1.0:
        align_gate = 1.0

    # ---------- 货箱速度 ----------
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---------- 完成条件（显式推断） ----------
    in_dock = (dx <= 0.024 and dx >= -0.024 and dy <= 0.030 and dy >= -0.030)
    aligned = (align_gate >= 0.75)  # cos(2θ) >= 0.75  => 误差约 < 30°
    slow = (crate_speed < 0.05)
    complete_now = in_dock and aligned and slow

    if complete_now:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- 组件 ----------
    components = {}

    # --- 主进度：增量式靠近（只在更接近时给分） ---
    progress = dist_n_prev - dist_n
    if progress < 0.0:
        progress = 0.0
    # 用对齐门控，避免"歪着推进去"
    components["crate_to_dock_progress"] = 1.0 * progress * (0.3 + 0.7 * align_gate)

    # --- 接触轻柔度 ---
    crate_along_heading = cvx * float(obs[2]) + cvy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    components["soft_contact"] = -0.05 * contact * closing

    # --- 接近泊位时的高速惩罚（只在未完成时生效，hinge） ---
    near_dock = 1.0 if dist_n < 3.0 else 0.0
    speed_excess = crate_speed - 0.05
    if speed_excess < 0.0:
        speed_excess = 0.0
    sp_pen = -0.5 * near_dock * (1.0 - contact) * speed_excess
    if complete_now:
        sp_pen = 0.0
    components["crate_speed_near_dock"] = sp_pen

    # --- 越界惩罚（小车 / 货箱），用已声明观测近似 ---
    oob = 0.0
    ax = float(next_obs[0])
    ay = float(next_obs[1])
    if ax > 0.95:
        oob += (ax - 0.95)
    if ax < -0.95:
        oob += (-0.95 - ax)
    if ay > 0.95:
        oob += (ay - 0.95)
    if ay < -0.95:
        oob += (-0.95 - ay)
    # 货箱世界坐标近似：小车位置 + 车体系偏移旋转
    ch = float(next_obs[2])
    sh = float(next_obs[3])
    relx = float(next_obs[6]) * 3.0
    rely = float(next_obs[7]) * 3.0
    crate_wx = ax * 5.0 + (relx * ch - rely * sh)
    crate_wy = ay * 4.0 + (relx * sh + rely * ch)
    if crate_wx > 4.7:
        oob += (crate_wx - 4.7) * 0.5
    if crate_wx < -4.7:
        oob += (-4.7 - crate_wx) * 0.5
    if crate_wy > 3.7:
        oob += (crate_wy - 3.7) * 0.5
    if crate_wy < -3.7:
        oob += (-3.7 - crate_wy) * 0.5
    components["out_of_bounds"] = -1.0 * oob

    # --- 障碍接近惩罚（前/左/右） ---
    obs_pen = 0.0
    sf = float(next_obs[15])
    sl = float(next_obs[16])
    sr = float(next_obs[17])
    if sf > 0.8:
        obs_pen += (sf - 0.8)
    if sl > 0.8:
        obs_pen += (sl - 0.8)
    if sr > 0.8:
        obs_pen += (sr - 0.8)
    components["obstacle"] = -0.3 * obs_pen

    # --- 一次性：首次进入泊位 ---
    enter_bonus = 0.0
    if in_dock and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_bonus = 5.0
    components["enter_dock_once"] = enter_bonus

    # --- 一次性：完成事件 ---
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["success_event"] = success_event

    # ---------- 完成状态下除一次性事件外必须为 0 ----------
    if complete_now:
        components["crate_to_dock_progress"] = 0.0
        components["soft_contact"] = 0.0
        components["crate_speed_near_dock"] = 0.0
        components["obstacle"] = 0.0
        components["out_of_bounds"] = 0.0
        # enter_dock_once 与 success_event 保留（一次性）

    total = 0.0
    for k in components:
        total += components[k]

    return (float(total), components)
```