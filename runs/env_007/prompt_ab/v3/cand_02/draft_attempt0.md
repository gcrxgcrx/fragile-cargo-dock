```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    _PREV_T = [-1.0]
    _STREAK = [0]
    _PAID = [False]
    _ENTERED = [False]

    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    components = {}

    # ---------- 几何量（由 obs 声明含义恢复） ----------
    dx = float(next_obs[12]) * 5.0      # 货箱到泊位 x 偏移 (m)
    dy = float(next_obs[13]) * 4.0      # 货箱到泊位 y 偏移 (m)
    dist = (dx * dx + dy * dy) ** 0.5

    odx = float(obs[12]) * 5.0
    ody = float(obs[13]) * 4.0
    odist = (odx * odx + ody * ody) ** 0.5

    # ---------- 1. 货箱向泊位推进（增量形式，避免悬停收割） ----------
    progress = odist - dist
    if progress > 0.05:
        progress = 0.05
    elif progress < -0.05:
        progress = -0.05
    components["crate_to_dock_progress"] = 12.0 * progress

    # ---------- 2. 货箱朝向对齐（仅在接近泊位时轻微引导） ----------
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    crate_ang = crate_sin
    if crate_cos < 0.0:
        crate_ang = 1.0 if crate_sin >= 0.0 else -1.0
    align_err = abs(crate_ang)
    if align_err > 1.0:
        align_err = 1.0
    near_gate = 1.0 / (1.0 + 6.0 * dist)
    components["crate_dock_alignment"] = 1.5 * (1.0 - align_err) * near_gate

    # ---------- 3. 接近泊位时的速度抑制（门控，仅在泊位附近激活） ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    if crate_speed > 3.0:
        crate_speed = 3.0
    dock_proximity = 1.0 / (1.0 + 10.0 * dist)
    components["crate_speed_near_dock"] = -0.8 * crate_speed * dock_proximity

    # ---------- 4. 接触轻柔度（本环境核心技能信号） ----------
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    components["soft_contact"] = -0.05 * contact * closing

    # ---------- 5. 越界防护（hinge，仅接近边界时生效） ----------
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    oob = 0.0
    if cart_x > 0.85:
        oob = oob + (cart_x - 0.85)
    elif cart_x < -0.85:
        oob = oob + (-0.85 - cart_x)
    if cart_y > 0.85:
        oob = oob + (cart_y - 0.85)
    elif cart_y < -0.85:
        oob = oob + (-0.85 - cart_y)
    components["out_of_bounds"] = -3.0 * oob

    # ---------- 6. 完成事件：一次性 ----------
    inside = 1.0 if (abs(dx) <= 0.30 and abs(dy) <= 0.30) else 0.0
    aligned = 1.0 if align_err < 0.5236 else 0.0
    still = 1.0 if crate_speed < 0.05 else 0.0

    if inside > 0.5 and aligned > 0.5 and still > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    enter_event = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 20.0

    components["success_event"] = success_event
    components["enter_dock_event"] = enter_event

    total = 0.0
    for key in components:
        total = total + components[key]

    return (float(total), components)
```