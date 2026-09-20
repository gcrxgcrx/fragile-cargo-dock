```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 回合边界检测（模块级状态） ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = -1.0
    _PREV_T[0] = t

    # ---------- 货箱到泊位距离（由 obs[12], obs[13] 恢复） ----------
    # 仓库半宽 ~5.0 m, 半高 ~4.0 m（由 obs[0]*5.0, obs[1]*4.0 推断）
    dx = float(next_obs[12]) * 5.0
    dy = float(next_obs[13]) * 4.0
    dist = (dx * dx + dy * dy) ** 0.5

    prev_dx = float(obs[12]) * 5.0
    prev_dy = float(obs[13]) * 4.0
    prev_dist = (prev_dx * prev_dx + prev_dy * prev_dy) ** 0.5

    # ---------- 完成条件（严格取自环境事实） ----------
    inside = 1.0 if (abs(float(next_obs[12])) <= 0.024 and abs(float(next_obs[13])) <= 0.030) else 0.0

    # 朝向误差：货箱朝向 vs 泊位期望朝向（泊位朝向取沿 +x，即 cos=1, sin=0）
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    # 朝向误差角（弧度），用 cos 分量近似 |sin| 作为误差度量
    heading_err_deg = 0.0
    # atan2 的近似：用 |sin| 与 cos 符号判断；30° 阈值对应 sin(30°)=0.5
    abs_sin = abs(crate_sin)
    heading_ok = 1.0 if (crate_cos > 0.0 and abs_sin < 0.5) else 0.0

    # 货箱速度（m/s）
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    speed_ok = 1.0 if crate_speed < 0.05 else 0.0

    # 连续保持计数
    if inside > 0.5 and heading_ok > 0.5 and speed_ok > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- 主信号：货箱向泊位推进（增量形式） ----------
    progress = prev_dist - dist
    if progress > 0.0:
        crate_progress = 1.0 * progress
    else:
        crate_progress = 1.0 * progress  # 允许负值，方向性引导

    # ---------- 接近泊位时的减速引导（仅接近时启用，门控） ----------
    near_gate = 1.0 if dist < 1.0 else 0.0
    speed_near_dock = -0.5 * near_gate * crate_speed

    # ---------- 朝向对齐 shaping（接近泊位时启用） ----------
    align_gate = 1.0 if dist < 1.5 else 0.0
    align_reward = 0.3 * align_gate * (1.0 - abs_sin if crate_cos > 0.0 else -1.0)

    # ---------- 接触轻柔度（核心技能信号） ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    cart_cos_h = float(obs[2])
    cart_sin_h = float(obs[3])
    crate_along_heading = crate_vx * cart_cos_h + crate_vy * cart_sin_h
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---------- 越界惩罚（hinge，仅接近边界时生效） ----------
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    oob = 0.0
    if abs(cart_x) > 0.85:
        oob -= 0.5 * (abs(cart_x) - 0.85)
    if abs(cart_y) > 0.85:
        oob -= 0.5 * (abs(cart_y) - 0.85)

    # ---------- 一次性事件 ----------
    # 首次进入泊位（整局一次）
    enter_event = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 50.0

    # 完成事件（连续 10 步满足，整局一次）
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # ---------- 汇总 ----------
    components = {}
    components["crate_progress"] = float(crate_progress)
    components["speed_near_dock"] = float(speed_near_dock)
    components["align_reward"] = float(align_reward)
    components["gentleness"] = float(gentleness)
    components["out_of_bounds"] = float(oob)
    components["enter_dock_event"] = float(enter_event)
    components["success_event"] = float(success_event)

    total_reward = (
        crate_progress
        + speed_near_dock
        + align_reward
        + gentleness
        + oob
        + enter_event
        + success_event
    )

    _PREV_DIST[0] = dist
    return (float(total_reward), components)
```