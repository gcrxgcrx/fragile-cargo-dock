```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 模块级状态：回合边界检测 + 一次性事件 ----------
    _PREV_T = [-1.0]
    _STREAK = [0]
    _PAID = [False]
    _ENTERED = [False]

    # 回合边界：obs[18] 单调递增，重置时回落
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---------- 泊位几何（来自环境事实） ----------
    # 货箱中心到泊位中心的有符号偏移（归一化）
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    # 容差：|dx| <= 0.024, |dy| <= 0.030 视为"完全进入"
    in_dock = (abs(dx) <= 0.024) and (abs(dy) <= 0.030)

    # ---------- 货箱速度 ----------
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    slow = crate_speed < 0.05

    # ---------- 朝向对齐 ----------
    # 货箱朝向误差
    cc = float(next_obs[10])
    cs = float(next_obs[11])
    crate_ang = 0.0
    if cc != 0.0 or cs != 0.0:
        # 近似 atan2 的连续量：用 cos 作为对齐度（0 度 -> 1.0）
        crate_ang = cc  # 货箱朝向余弦，对齐时接近 1（假设泊位朝向为 0）
    align_ok = crate_ang > 0.866  # cos(30 deg)

    # ---------- 完成条件的连续计数（一次性事件） ----------
    complete_now = in_dock and slow and align_ok
    if complete_now:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # 首次进入泊位一次性奖励
    entered_event = 0.0
    if in_dock and not _ENTERED[0]:
        _ENTERED[0] = True
        entered_event = 30.0

    # ---------- 组件 1：货箱向泊位的增量进展 ----------
    # 使用增量形式，避免悬停收割
    prev_dx = float(obs[12])
    prev_dy = float(obs[13])
    prev_dist = (prev_dx * prev_dx + prev_dy * prev_dy) ** 0.5
    next_dist = (dx * dx + dy * dy) ** 0.5
    progress = prev_dist - next_dist  # 正值表示更接近
    # 限制单步幅度，避免异常跳变
    if progress > 0.05:
        progress = 0.05
    if progress < -0.05:
        progress = -0.05
    crate_to_dock_progress = 5.0 * progress

    # ---------- 组件 2：接近泊位时的速度抑制（仅在接近时激活） ----------
    # 距离泊位中心（归一化）
    dock_dist = (dx * dx + dy * dy) ** 0.5
    # 仅在接近泊位时启用（例如归一化距离 < 0.15）
    near_dock_factor = 0.0
    if dock_dist < 0.15:
        near_dock_factor = 1.0 - dock_dist / 0.15
        if near_dock_factor < 0.0:
            near_dock_factor = 0.0
    # 速度超过阈值时惩罚（hinge 形式）
    speed_excess = crate_speed - 0.05
    if speed_excess < 0.0:
        speed_excess = 0.0
    crate_speed_penalty_near_dock = -0.5 * near_dock_factor * speed_excess

    # ---------- 组件 3：接触轻柔度（核心技能信号） ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    cart_cos = float(obs[2])
    cart_sin = float(obs[3])
    crate_along_heading = crate_vx * cart_cos + crate_vy * cart_sin
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---------- 组件 4：朝向对齐 shaping（接近泊位时激活） ----------
    # 只在接近泊位时引导朝向对齐
    align_shaping = 0.0
    if near_dock_factor > 0.0:
        align_error = 1.0 - crate_ang  # 0 表示对齐
        if align_error < 0.0:
            align_error = 0.0
        align_shaping = 0.3 * near_dock_factor * (1.0 - align_error)

    # ---------- 组件 5：越界惩罚（hinge 形式） ----------
    # 小车位置越界（obs[0], obs[1] 归一化到 [-1,1] 附近）
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    oob_penalty = 0.0
    if abs(cart_x) > 0.95:
        oob_penalty -= 1.0 * (abs(cart_x) - 0.95)
    if abs(cart_y) > 0.95:
        oob_penalty -= 1.0 * (abs(cart_y) - 0.95)
    # 货箱到泊位偏移过大也视为潜在越界风险
    if abs(dx) > 0.95:
        oob_penalty -= 1.0 * (abs(dx) - 0.95)
    if abs(dy) > 0.95:
        oob_penalty -= 1.0 * (abs(dy) - 0.95)

    # ---------- 组件 6：动作平滑（轻量） ----------
    a0 = float(action[0])
    a1 = float(action[1])
    action_smoothness = -0.01 * (a0 * a0 + a1 * a1)

    # ---------- 汇总 ----------
    components = {}
    components["crate_to_dock_progress"] = crate_to_dock_progress
    components["crate_speed_penalty_near_dock"] = crate_speed_penalty_near_dock
    components["gentleness"] = gentleness
    components["align_shaping"] = align_shaping
    components["out_of_bounds_penalty"] = oob_penalty
    components["action_smoothness"] = action_smoothness
    components["entered_dock_event"] = entered_event
    components["success_event"] = success_event

    total_reward = (
        crate_to_dock_progress
        + crate_speed_penalty_near_dock
        + gentleness
        + align_shaping
        + oob_penalty
        + action_smoothness
        + entered_event
        + success_event
    )

    return (float(total_reward), components)
```