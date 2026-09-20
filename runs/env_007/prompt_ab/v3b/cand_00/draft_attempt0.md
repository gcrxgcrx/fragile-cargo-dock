```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_DIST = [-1.0]
_PREV_ALIGN = [-1.0]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode 边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = -1.0
        _PREV_ALIGN[0] = -1.0
    _PREV_T[0] = t

    components = {}

    # ---------- 几何量（泊位容差：|dx|<=0.024, |dy|<=0.030） ----------
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    # 货箱朝向误差（弧度）
    cch = float(next_obs[10])
    csh = float(next_obs[11])
    ang_err = abs(__import__('math').atan2(csh, cch)) if False else abs(_atan2(csh, cch))

    # 货箱速度（世界系，m/s）
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---------- 主信号 1：货箱到泊位的增量式进度 ----------
    # 只在"这一帧更接近了"时给分，避免悬停收割
    progress = 0.0
    if _PREV_DIST[0] >= 0.0:
        progress = _PREV_DIST[0] - dist
        if progress > 0.5:
            progress = 0.5
        if progress < -0.5:
            progress = -0.5
    _PREV_DIST[0] = dist
    components["crate_to_dock_progress"] = 8.0 * progress

    # ---------- 主信号 2：朝向对齐的增量式进度 ----------
    align_prog = 0.0
    if _PREV_ALIGN[0] >= 0.0:
        align_prog = _PREV_ALIGN[0] - ang_err
        if align_prog > 0.5:
            align_prog = 0.5
        if align_prog < -0.5:
            align_prog = -0.5
    _PREV_ALIGN[0] = ang_err
    components["crate_align_progress"] = 2.0 * align_prog

    # ---------- 接近泊位时的速度抑制（门控：只在接近泊位时激活） ----------
    near = 0.0
    if dist < 0.20:
        near = (0.20 - dist) / 0.20
    if near > 1.0:
        near = 1.0
    speed_excess = crate_speed - 0.05
    if speed_excess < 0.0:
        speed_excess = 0.0
    components["crate_speed_penalty_near_dock"] = -2.0 * near * (speed_excess ** 2)

    # ---------- 接触轻柔度（唯一能教会"接近泊位减速"的信号） ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    components["soft_contact_penalty"] = -0.05 * contact * closing

    # ---------- 越界防护（hinge：只在贴近边界时生效） ----------
    cart_x = abs(float(next_obs[0]))
    cart_y = abs(float(next_obs[1]))
    oob_cart = 0.0
    if cart_x > 0.90:
        oob_cart += (cart_x - 0.90) ** 2
    if cart_y > 0.90:
        oob_cart += (cart_y - 0.90) ** 2
    components["cart_out_of_bounds_penalty"] = -5.0 * oob_cart

    # 货箱越界近似：泊位偏移的绝对值过大（货箱远离泊位且接近仓库外沿）
    crate_oob = 0.0
    if abs(dx) > 0.85:
        crate_oob += (abs(dx) - 0.85) ** 2
    if abs(dy) > 0.85:
        crate_oob += (abs(dy) - 0.85) ** 2
    components["crate_out_of_bounds_penalty"] = -5.0 * crate_oob

    # ---------- 静态障碍接近惩罚（hinge） ----------
    obs_front = float(next_obs[15])
    obs_left = float(next_obs[16])
    obs_right = float(next_obs[17])
    obs_pen = 0.0
    if obs_front > 0.85:
        obs_pen += (obs_front - 0.85) ** 2
    if obs_left > 0.85:
        obs_pen += (obs_left - 0.85) ** 2
    if obs_right > 0.85:
        obs_pen += (obs_right - 0.85) ** 2
    components["obstacle_proximity_penalty"] = -3.0 * obs_pen

    # ---------- 动作平滑（轻量，防止抖动） ----------
    a0 = float(action[0])
    a1 = float(action[1])
    components["action_smoothness"] = -0.01 * (a0 * a0 + a1 * a1)

    # ---------- 完成条件（显式从 obs 推断，容差取自环境事实） ----------
    inside = (abs(dx) <= 0.024) and (abs(dy) <= 0.030)
    aligned = (ang_err < 0.5235987755982988)  # 30 度
    slow = (crate_speed < 0.05)
    done_cond = inside and aligned and slow

    if done_cond:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # 首次进入泊位（一次性）
    enter_event = 0.0
    if inside and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 30.0
    components["first_enter_dock_bonus"] = enter_event

    # 完成事件（一次性，整局只发一次）
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["success_event_bonus"] = success_event

    total = 0.0
    for k in components:
        total += components[k]

    return (float(total), components)


def _atan2(y, x):
    # 无 import 的 atan2 近似实现（足够用于角度误差判断）
    if x == 0.0 and y == 0.0:
        return 0.0
    ax = x if x >= 0.0 else -x
    ay = y if y >= 0.0 else -y
    if ax >= ay:
        r = ay / (ax + 1e-12)
        # atan(r) 泰勒/多项式近似
        s = r / (1.0 + 0.28086 * r * r)
        a = s
    else:
        r = ax / (ay + 1e-12)
        s = r / (1.0 + 0.28086 * r * r)
        a = 1.5707963267948966 - s
    if x < 0.0:
        a = 3.141592653589793 - a
    if y < 0.0:
        a = -a
    return a
```