# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 回合边界检测（obs[18] 单调递增，重置时回落） ----
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = -1.0
    _PREV_T[0] = t

    # ---- 几何量：货箱到泊位中心的偏移（归一化） ----
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    # 泊位容差（来自环境事实）：|dx|<=0.024, |dy|<=0.030
    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0

    # ---- 朝向误差（货箱朝向 vs 泊位朝向 0 度） ----
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    # 朝向误差角（弧度），用 cos 近似
    heading_err = 1.0 - ch          # 0 表示完全对齐，最大 2
    aligned = 1.0 if ch > 0.866 else 0.0   # 30 度容差 (cos30≈0.866)

    # ---- 货箱速度（世界系，m/s） ----
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    slow = 1.0 if crate_speed < 0.05 else 0.0

    components = {}

    # ================= 1. 主推进信号（增量形式，避免悬停收割） =================
    # 用上一帧距离做 delta，只在"这一帧更接近"时给分
    if _PREV_DIST[0] < 0.0:
        _PREV_DIST[0] = dist
    progress = _PREV_DIST[0] - dist          # >0 表示更接近泊位
    if progress < 0.0:
        progress = progress * 0.5            # 远离时轻微惩罚，但不压过推进
    components["crate_to_dock_progress"] = 8.0 * progress
    _PREV_DIST[0] = dist

    # ================= 2. 朝向对齐 shaping（增量/门控形式） =================
    # 只在货箱接近泊位时才给对齐信号，避免全局收分
    near_gate = 1.0
    if dist > 0.15:
        near_gate = max(0.0, 1.0 - (dist - 0.15) / 0.35)
    align_bonus = near_gate * (0.3 * ch)      # ch 越大越对齐
    components["crate_heading_align"] = 0.5 * align_bonus

    # ================= 3. 接近泊位时的速度抑制（门控，仅在泊位附近激活） =================
    # 只在货箱已接近泊位（dist<0.10）时激活，不会阻碍到达
    speed_gate = 0.0
    if dist < 0.10:
        speed_gate = 1.0 - dist / 0.10
    speed_pen = -2.0 * speed_gate * crate_speed
    components["crate_speed_near_dock"] = speed_pen

    # ================= 4. 接触轻柔度（核心技能信号） =================
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing
    components["soft_contact_penalty"] = gentleness

    # ================= 5. 越界惩罚（hinge，只在边界附近生效） =================
    cx = float(next_obs[0])
    cy = float(next_obs[1])
    ob_pen = 0.0
    if abs(cx) > 0.85:
        ob_pen -= 1.0 * (abs(cx) - 0.85)
    if abs(cy) > 0.85:
        ob_pen -= 1.0 * (abs(cy) - 0.85)
    components["out_of_bounds_penalty"] = ob_pen

    # ================= 6. 障碍接近惩罚（hinge，轻量） =================
    sf = float(next_obs[15])
    obs_pen = 0.0
    if sf > 0.9:
        obs_pen -= 0.5 * (sf - 0.9)
    components["obstacle_penalty"] = obs_pen

    # ================= 7. 完成事件（一次性，连续 10 步） =================
    if inside > 0.5 and aligned > 0.5 and slow > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["success_event"] = success_event

    # 首次进入泊位的一次性奖励
    entered_bonus = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        entered_bonus = 20.0
    components["first_enter_dock"] = entered_bonus

    total_reward = 0.0
    for k in components:
        total_reward += components[k]

    return (float(total_reward), components)
```
