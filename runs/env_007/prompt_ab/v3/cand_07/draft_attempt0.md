```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 回合边界检测（模块级状态，obs[18] 单调递增，重置时回落） ----------
    _PREV_T = compute_reward._PREV_T
    _STREAK = compute_reward._STREAK
    _PAID = compute_reward._PAID
    _ENTERED = compute_reward._ENTERED

    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---------- 泊位几何（容差取自环境事实） ----------
    # 货箱中心到泊位中心的有符号偏移（归一化）
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    # 恢复米制偏移：x 用仓库半宽 5.0，y 用仓库半高 4.0
    off_x = dx * 5.0
    off_y = dy * 4.0
    dist_dock = (off_x * off_x + off_y * off_y) ** 0.5

    # 完成判据：|obs[12]|<=0.024 且 |obs[13]|<=0.030，朝向误差<30°，速度<0.05 m/s
    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0

    # 货箱速度（世界系，m/s）
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向误差（弧度），泊位朝向隐含与仓库轴对齐
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    # 朝向误差 = atan2 的绝对值；用 cos 值近似，误差 < 30° => cos > cos(30°)=0.866
    heading_cos = ch  # 货箱朝向余弦（相对世界 x 轴）
    heading_ok = 1.0 if heading_cos >= 0.866 else 0.0

    # ---------- 组件 1：货箱到泊位的增量推进（improvement_delta） ----------
    pdx = float(obs[12])
    pdy = float(obs[13])
    prev_dist = ((pdx * 5.0) ** 2 + (pdy * 4.0) ** 2) ** 0.5
    progress = prev_dist - dist_dock  # 正=更接近
    crate_progress = 1.0 * progress

    # ---------- 组件 2：接近泊位时的货箱速度抑制（仅在接近泊位时激活） ----------
    # 用门控：离泊位越近，速度惩罚越强；远离时不惩罚，避免阻碍到达
    near_gate = 1.0 / (1.0 + 2.0 * dist_dock)  # dist_dock 小 -> gate 接近 1
    crate_speed_penalty = -0.30 * near_gate * (crate_speed ** 2)

    # ---------- 组件 3：接触轻柔度（唯一能教减速的信号） ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---------- 组件 4：朝向对齐 shaping（仅在接近泊位时激活） ----------
    heading_err = 1.0 - heading_cos  # 0=完美对齐
    if heading_err < 0.0:
        heading_err = 0.0
    heading_penalty = -0.20 * near_gate * heading_err

    # ---------- 组件 5：越界轻量惩罚（hinge，边界附近才生效） ----------
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    oob = 0.0
    if abs(cart_x) > 0.95:
        oob += (abs(cart_x) - 0.95)
    if abs(cart_y) > 0.95:
        oob += (abs(cart_y) - 0.95)
    # 货箱越界（由偏移近似仓库范围，货箱世界坐标较难精确，用 dx/dy 与小车位置共同约束）
    if abs(dx) > 1.0:
        oob += (abs(dx) - 1.0)
    if abs(dy) > 1.0:
        oob += (abs(dy) - 1.0)
    oob_penalty = -1.0 * oob

    # ---------- 组件 6：动作平滑（轻量，避免剧烈抖动） ----------
    a0 = float(action[0])
    a1 = float(action[1])
    smooth_penalty = -0.01 * (a0 * a0 + a1 * a1)

    # ---------- 完成事件：一次性 ----------
    done_cond = (inside > 0.5) and (heading_ok > 0.5) and (crate_speed < 0.05)
    if done_cond:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # 首次进入泊位的一次性奖励
    entered_event = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        entered_event = 30.0

    total = (
        crate_progress
        + crate_speed_penalty
        + gentleness
        + heading_penalty
        + oob_penalty
        + smooth_penalty
        + success_event
        + entered_event
    )

    components = {
        "crate_progress": crate_progress,
        "crate_speed_penalty": crate_speed_penalty,
        "gentleness": gentleness,
        "heading_penalty": heading_penalty,
        "oob_penalty": oob_penalty,
        "smooth_penalty": smooth_penalty,
        "success_event": success_event,
        "entered_event": entered_event,
    }
    return float(total), components


# 模块级状态初始化（挂在函数属性上，避免 global 语句）
compute_reward._PREV_T = [-1.0]
compute_reward._STREAK = [0]
compute_reward._PAID = [False]
compute_reward._ENTERED = [False]
```