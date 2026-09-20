```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- episode 边界检测（obs[18] 单调递增，重置时回落） ----
    if not hasattr(compute_reward, "_state"):
        compute_reward._state = {
            "prev_t": -1.0,
            "streak": 0,
            "paid": False,
            "entered": False,
            "prev_dist": None,
        }
    st = compute_reward._state

    t = float(next_obs[18])
    if t < st["prev_t"] or t <= 1.0 / 400.0:
        st["streak"] = 0
        st["paid"] = False
        st["entered"] = False
        st["prev_dist"] = None
    st["prev_t"] = t

    # ---- 泊位几何（来自环境事实：半宽 5.0 m? 用声明阈值） ----
    # |obs[12]| <= 0.024, |obs[13]| <= 0.030 表示完全进入泊位
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    # ---- 朝向误差 ----
    crate_ang = 0.0
    if not (next_obs[10] == 0.0 and next_obs[11] == 0.0):
        crate_ang = (next_obs[11] * next_obs[11]) ** 0.5  # 近似 |sin|
    # 用 cos 分量衡量对齐：|cos| 越接近 1 越对齐
    cos_align = abs(float(next_obs[10]))
    # 朝向误差 < 30° => cos > cos(30°)=0.866
    align_ok = 1.0 if cos_align >= 0.866 else 0.0

    # ---- 货箱速度 ----
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    speed_ok = 1.0 if crate_speed < 0.05 else 0.0

    # ---- 完成条件 ----
    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0
    done_cond = 1.0 if (inside > 0.5 and align_ok > 0.5 and speed_ok > 0.5) else 0.0
    if done_cond > 0.5:
        st["streak"] += 1
    else:
        st["streak"] = 0

    components = {}

    # ---- 1) 增量接近信号（只在更接近时给分，避免悬停收割） ----
    progress = 0.0
    if st["prev_dist"] is not None:
        progress = st["prev_dist"] - dist
    st["prev_dist"] = dist
    # 只在推进时给正分，远离时给轻微负分
    progress_reward = 6.0 * progress
    if progress_reward > 0.6:
        progress_reward = 0.6
    if progress_reward < -0.6:
        progress_reward = -0.6
    components["crate_progress"] = progress_reward

    # ---- 2) 轻柔接触惩罚（接触且接近时） ----
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing
    components["soft_contact"] = gentleness

    # ---- 3) 接近泊位时的货箱速度抑制（门控：仅在靠近泊位时激活） ----
    near_dock = 0.0
    if dist < 0.30:
        near_dock = 1.0 - dist / 0.30
        if near_dock < 0.0:
            near_dock = 0.0
    speed_penalty = -0.4 * near_dock * (crate_speed ** 2)
    components["dock_speed_penalty"] = speed_penalty

    # ---- 4) 朝向对齐 shaping（仅在靠近泊位时激活） ----
    align_shaping = 0.0
    if dist < 0.30:
        align_shaping = 0.3 * near_dock * (cos_align - 0.5)
        if align_shaping < 0.0:
            align_shaping = 0.0
    components["align_shaping"] = align_shaping

    # ---- 5) 越界惩罚（小车与货箱） ----
    oob = 0.0
    cx = float(next_obs[0])
    cy = float(next_obs[1])
    if abs(cx) > 0.95:
        oob -= 0.5 * (abs(cx) - 0.95)
    if abs(cy) > 0.95:
        oob -= 0.5 * (abs(cy) - 0.95)
    # 货箱世界坐标
    cc = float(next_obs[2])
    cs = float(next_obs[3])
    relx = float(next_obs[6]) * 3.0
    rely = float(next_obs[7]) * 3.0
    crate_wx = cx * 5.0 + relx * cc - rely * cs
    crate_wy = cy * 4.0 + relx * cs + rely * cc
    if abs(crate_wx) > 4.7:
        oob -= 0.5 * (abs(crate_wx) - 4.7)
    if abs(crate_wy) > 3.7:
        oob -= 0.5 * (abs(crate_wy) - 3.7)
    if oob < -1.0:
        oob = -1.0
    components["out_of_bounds"] = oob

    # ---- 6) 动作平滑（轻量，不压制推动） ----
    smooth = -0.02 * (float(action[0]) ** 2) - 0.01 * (float(action[1]) ** 2)
    components["action_smooth"] = smooth

    # ---- 7) 首次进入泊位（一次性） ----
    if inside > 0.5 and not st["entered"]:
        st["entered"] = True
        components["first_enter_bonus"] = 15.0
    else:
        components["first_enter_bonus"] = 0.0

    # ---- 8) 完成事件（一次性，连续 10 步） ----
    success_event = 0.0
    if st["streak"] >= 10 and not st["paid"]:
        st["paid"] = True
        success_event = 300.0
    components["success_event"] = success_event

    total = 0.0
    for k in components:
        total += components[k]
    return (float(total), components)
```