```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- episode 边界检测 ----
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---- 货箱到泊位偏移（归一化 -> 米）----
    # 仓库半宽 ~ 5.0 m, 半高 ~ 4.0 m（由 obs 缩放反推）
    dx = float(next_obs[12]) * 5.0
    dy = float(next_obs[13]) * 4.0
    dist = (dx * dx + dy * dy) ** 0.5

    odx = float(obs[12]) * 5.0
    ody = float(obs[13]) * 4.0
    odist = (odx * odx + ody * ody) ** 0.5

    # ---- 完成判据（严格取自环境事实）----
    # 容差：|obs[12]| <= 0.024 且 |obs[13]| <= 0.030，朝向误差 < 30°，速度 < 0.05 m/s
    in_tol = (abs(float(next_obs[12])) <= 0.024) and (abs(float(next_obs[13])) <= 0.030)

    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    # 朝向误差：货箱朝向 vs 泊位朝向（泊位朝向假定为 0，即 cos=1,sin=0）
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    # 归一化朝向向量
    cn = (crate_cos * crate_cos + crate_sin * crate_sin) ** 0.5
    if cn > 1e-6:
        crate_cos = crate_cos / cn
        crate_sin = crate_sin / cn
    # 与 (1,0) 的夹角误差
    align_cos = crate_cos  # cos(theta_error)
    align_error_deg = 0.0
    if align_cos > 1.0:
        align_cos = 1.0
    if align_cos < -1.0:
        align_cos = -1.0
    # 用 acos 近似：acos(x) ~ sqrt(2*(1-x)) 在 x 接近 1 时可用
    # 但这里直接判 cos >= cos(30°)=0.866
    aligned = align_cos >= 0.866

    slow = crate_speed < 0.05

    completed_now = in_tol and aligned and slow

    if completed_now:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    components = {}

    # ================= 完成事件（一次性）=================
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["success_event"] = success_event

    # 首次进入泊位（一次性）
    entered_bonus = 0.0
    if in_tol and not _ENTERED[0]:
        _ENTERED[0] = True
        entered_bonus = 30.0
    components["first_entered_dock"] = entered_bonus

    # ================= 完成状态下其余组件必须为 0 =================
    if completed_now:
        # 除一次性事件外，其余全部为 0
        total = success_event + entered_bonus
        return (float(total), components)

    # ================= 主进度信号：增量靠近 =================
    # 只在"这一帧更接近了"时给分，且用对齐度做门控
    progress = odist - dist
    if progress < 0.0:
        progress = 0.0
    # 对齐门控（货箱朝向与泊位朝向接近时才放大推进收益）
    align_gate = 0.5 + 0.5 * max(0.0, align_cos)
    # 限制单步增量上限，避免瞬移刷分
    if progress > 0.5:
        progress = 0.5
    crate_progress_reward = 8.0 * progress * align_gate
    components["crate_to_dock_progress"] = crate_progress_reward

    # ================= 轻柔接触（核心技能）=================
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    # k 加大到 2.0，使 1.0 m/s 撞击时惩罚 ~2.0，与推进项同量级或更大
    gentleness = -2.0 * contact * closing
    components["soft_contact"] = gentleness

    # ================= 接近泊位时的速度抑制（hinge，仅惩罚）=================
    # 仅在货箱已接近泊位（dist < 1.5 m）且速度过快时扣分
    speed_penalty = 0.0
    if dist < 1.5:
        # 越靠近泊位，允许速度越低
        if dist < 0.5:
            allowed = 0.3
        else:
            allowed = 0.3 + (dist - 0.5) * 1.0
        if crate_speed > allowed:
            speed_penalty = -3.0 * (crate_speed - allowed)
    components["crate_speed_near_dock"] = speed_penalty

    # ================= 越界守卫 =================
    # 小车边界：|obs[0]| 或 |obs[1]| 超过 0.95 时明显惩罚
    cx = abs(float(next_obs[0]))
    cy = abs(float(next_obs[1]))
    bound_pen = 0.0
    margin = 0.95
    if cx > margin:
        bound_pen -= 20.0 * (cx - margin)
    if cy > margin:
        bound_pen -= 20.0 * (cy - margin)
    # 硬墙外（>1.05）额外重罚
    if cx > 1.05:
        bound_pen -= 50.0 * (cx - 1.05)
    if cy > 1.05:
        bound_pen -= 50.0 * (cy - 1.05)
    components["out_of_bounds"] = bound_pen

    # ================= 货箱越界守卫 =================
    # 货箱世界坐标恢复：车体系 -> 世界系
    rel_x = float(next_obs[6]) * 3.0
    rel_y = float(next_obs[7]) * 3.0
    ch = float(next_obs[2])
    sh = float(next_obs[3])
    crate_wx = float(next_obs[0]) * 5.0 + rel_x * ch - rel_y * sh
    crate_wy = float(next_obs[1]) * 4.0 + rel_x * sh + rel_y * ch
    crate_bx = abs(crate_wx) / 5.0
    crate_by = abs(crate_wy) / 4.0
    crate_bound_pen = 0.0
    if crate_bx > margin:
        crate_bound_pen -= 20.0 * (crate_bx - margin)
    if crate_by > margin:
        crate_bound_pen -= 20.0 * (crate_by - margin)
    if crate_bx > 1.05:
        crate_bound_pen -= 50.0 * (crate_bx - 1.05)
    if crate_by > 1.05:
        crate_bound_pen -= 50.0 * (crate_by - 1.05)
    components["crate_out_of_bounds"] = crate_bound_pen

    # ================= 轻量动作平滑 =================
    a0 = float(action[0])
    a1 = float(action[1])
    smooth_pen = -0.02 * (a1 * a1)
    components["action_smoothness"] = smooth_pen

    total = 0.0
    total += components["success_event"]
    total += components["first_entered_dock"]
    total += components["crate_to_dock_progress"]
    total += components["soft_contact"]
    total += components["crate_speed_near_dock"]
    total += components["out_of_bounds"]
    total += components["crate_out_of_bounds"]
    total += components["action_smoothness"]

    return (float(total), components)
```