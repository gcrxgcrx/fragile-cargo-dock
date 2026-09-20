# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    # 货箱到 dock 中心偏移（世界尺度：半宽 5.0，半高 4.0）
    dx = obs[12] * 5.0
    dy = obs[13] * 4.0
    ndx = next_obs[12] * 5.0
    ndy = next_obs[13] * 4.0

    dist = (dx * dx + dy * dy) ** 0.5
    ndist = (ndx * ndx + ndy * ndy) ** 0.5

    # 货箱速率（世界尺度 3.0 m/s 归一化）
    cvx = obs[8] * 3.0
    cvy = obs[9] * 3.0
    ncvx = next_obs[8] * 3.0
    ncvy = next_obs[9] * 3.0
    speed = (cvx * cvx + cvy * cvy) ** 0.5
    nspeed = (ncvx * ncvx + ncvy * ncvy) ** 0.5

    # 货箱朝向误差（相对 dock 对齐方向，dock 朝向按 0 处理）
    heading_err = abs(obs[11])
    # 归一化到 [0,1] 的连续对齐因子
    align_factor = 1.0 / (1.0 + 4.0 * heading_err)

    # 接触与小车速度
    contact = obs[14]
    cart_forward_speed = abs(obs[4] * 3.0)

    # 时间比例
    time_frac = obs[18]

    # ---------- 1. crate_to_dock_progress (主信号, delta 形式) ----------
    progress = dist - ndist
    # 凸化：距离越近，同样推进给更多奖励
    proximity = 1.0 / (1.0 + 1.0 * ndist)
    crate_to_dock_progress = 2.0 * progress * proximity

    # ---------- 2. crate_dock_alignment (接近 dock 时启用) ----------
    near_gate = 1.0 / (1.0 + 1.5 * ndist)
    crate_dock_alignment = 0.6 * near_gate * align_factor

    # ---------- 3. crate_settling (接近 dock 时抑制货箱速度) ----------
    # hinge：只在货箱接近 dock 且速度偏大时惩罚
    settle_gate = 1.0 / (1.0 + 2.0 * ndist)
    speed_excess = max(0.0, nspeed - 0.05)
    crate_settling = -1.5 * settle_gate * speed_excess

    # ---------- 4. contact_gated_push ----------
    # 接触且朝向 dock 推进时，鼓励有效推动
    push_dir = 0.0
    if ndist > 1e-6:
        push_dir = (dist - ndist) / (ndist + 1e-6)
    contact_gated_push = 0.5 * contact * max(0.0, push_dir) * (1.0 - near_gate * 0.0)

    # ---------- 5. impact_softness (接触时速度突变过大则惩罚) ----------
    # 用货箱速度突变近似冲击强度
    speed_jump = abs(nspeed - speed)
    impact = contact * max(0.0, speed_jump - 0.5) * cart_forward_speed
    impact_softness = -0.8 * min(impact, 3.0)

    # ---------- 6. boundary_avoidance ----------
    # 小车位置越界风险（|cart_x|>1 或 |cart_y|>1 即出界）
    cart_x = obs[0]
    cart_y = obs[1]
    cart_margin = max(0.0, abs(cart_x) - 0.85) + max(0.0, abs(cart_y) - 0.85)
    # 货箱世界坐标反推（粗略）：用 obs[6], obs[7] 车体系 + 小车位置
    crx = obs[6] * 3.0
    cry = obs[7] * 3.0
    crate_margin = max(0.0, abs(crx) - 2.5) + max(0.0, abs(cry) - 2.5)
    boundary_avoidance = -1.0 * (cart_margin + crate_margin)

    # ---------- 7. time_efficiency (轻度) ----------
    time_efficiency = -0.2 * time_frac * time_frac

    components = {
        "crate_to_dock_progress": float(crate_to_dock_progress),
        "crate_dock_alignment": float(crate_dock_alignment),
        "crate_settling": float(crate_settling),
        "contact_gated_push": float(contact_gated_push),
        "impact_softness": float(impact_softness),
        "boundary_avoidance": float(boundary_avoidance),
        "time_efficiency": float(time_efficiency),
    }

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return (float(total_reward), components)
```
