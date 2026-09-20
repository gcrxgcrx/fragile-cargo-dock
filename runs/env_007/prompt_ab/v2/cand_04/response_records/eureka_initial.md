# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 几何 / 阈值常量（来自环境事实）----
    # 泊位容差: |obs[12]| <= 0.024, |obs[13]| <= 0.030
    dock_tol_x = 0.024
    dock_tol_y = 0.030
    # 朝向误差阈值 30 度
    heading_tol = 0.5235987755982988  # 30 deg in rad
    # 速度阈值 0.05 m/s (obs[8], obs[9] 已除以 3.0)
    speed_tol = 0.05 / 3.0

    # ---- 当前 & 下一步货箱到泊位偏移（归一化）----
    dx_cur = obs[12]
    dy_cur = obs[13]
    dx_next = next_obs[12]
    dy_next = next_obs[13]

    # 归一化距离（用容差作为尺度，便于与阈值比较）
    dist_cur = ((dx_cur / dock_tol_x) ** 2 + (dy_cur / dock_tol_y) ** 2) ** 0.5
    dist_next = ((dx_next / dock_tol_x) ** 2 + (dy_next / dock_tol_y) ** 2) ** 0.5

    # ---- 货箱世界系速度（obs[8], obs[9] 为 /3.0 m/s）----
    vx = next_obs[8] * 3.0
    vy = next_obs[9] * 3.0
    crate_speed = (vx * vx + vy * vy) ** 0.5

    # ---- 货箱朝向误差 ----
    crate_ang = 0.0
    c_cos = next_obs[10]
    c_sin = next_obs[11]
    if c_cos != 0.0 or c_sin != 0.0:
        crate_ang = (c_sin ** 2) ** 0.5  # |sin| 近似朝向误差下界
    # 用 atan2 的等价形式：误差 = |asin(sin)| 简化，用 cos 更稳
    # 朝向误差 = acos(cos) 当 cos 接近 1
    heading_err = 0.0
    ccos = next_obs[10]
    if ccos > 1.0:
        ccos = 1.0
    if ccos < -1.0:
        ccos = -1.0
    # acos 近似: 用 |sin| + (1-cos) 组合估计，避免引入 math
    heading_err = (c_sin * c_sin) ** 0.5

    # ---- 完成条件代理（显式从 obs 推断，容差取自环境事实）----
    inside_dock = 1.0 if (abs(dx_next) <= dock_tol_x and abs(dy_next) <= dock_tol_y) else 0.0
    heading_ok = 1.0 if heading_err < (heading_tol * 0.5) else 0.0  # sin(30/2)≈0.26
    speed_ok = 1.0 if crate_speed < 0.05 else 0.0
    contact_flag = next_obs[14]

    # ============ 组件 1：货箱推进增量（主进度信号，增量形式）============
    # 只在"这一帧更接近泊位"时给分，避免悬停收割
    progress = dist_cur - dist_next  # 正=更接近
    if progress > 0.0:
        crate_progress_reward = 2.0 * progress
    else:
        crate_progress_reward = 0.0
    # 若货箱已在泊位内，不再给推进增量（防止在泊位内来回蹭）
    if inside_dock > 0.0:
        crate_progress_reward = 0.0

    # ============ 组件 2：货箱接近泊位的稀疏引导（接近度门控）============
    # 用 bounded 形式，但只在接近泊位时激活，且作为增量补充
    near_gate = max(0.0, 1.0 - dist_next / 20.0)  # 20 倍容差内开始激活
    crate_near_reward = 0.3 * near_gate

    # ============ 组件 3：泊位内对齐与静止（联合条件代理）============
    # 仅在货箱已进入泊位时激活
    if inside_dock > 0.0:
        align_factor = max(0.0, 1.0 - heading_err / heading_tol)
        speed_factor = max(0.0, 1.0 - crate_speed / 0.05)
        docking_quality = 1.5 * (align_factor * speed_factor) ** 0.5
    else:
        docking_quality = 0.0

    # ============ 组件 4：轻柔接触（仅在接触时轻罚速度突变）============
    # 用接触标志 + 货箱速度大小做软约束，避免误判正常推动
    soft_contact = 0.0
    if contact_flag > 0.5:
        # 货箱速度大时轻罚，抑制硬碰撞
        if crate_speed > 0.3:
            soft_contact = -0.5 * (crate_speed - 0.3)

    # ============ 组件 5：接近泊位时的速度抑制（门控，非全局）============
    # 仅在货箱接近泊位时启用，且货箱未进入泊位时
    if inside_dock <= 0.0 and dist_next < 5.0:
        if crate_speed > 0.1:
            speed_penalty = -0.4 * (crate_speed - 0.1)
        else:
            speed_penalty = 0.0
    else:
        speed_penalty = 0.0

    # ============ 组件 6：越界防护（hinge）============
    # 小车位置 obs[0], obs[1] 归一化，边界约 ±1.0
    out_penalty = 0.0
    cart_x = obs[0]
    cart_y = obs[1]
    if abs(cart_x) > 0.85:
        out_penalty -= 2.0 * (abs(cart_x) - 0.85)
    if abs(cart_y) > 0.85:
        out_penalty -= 2.0 * (abs(cart_y) - 0.85)
    # 货箱到泊位偏移过大（货箱可能越界）: 用 |dx|,|dy| 归一化
    if abs(dx_next) > 0.9:
        out_penalty -= 2.0 * (abs(dx_next) - 0.9)
    if abs(dy_next) > 0.9:
        out_penalty -= 2.0 * (abs(dy_next) - 0.9)

    # ============ 组件 7：完成事件奖励（主导过程信号）============
    # 完成条件：货箱在泊位内 + 朝向对齐 + 速度静止
    # 由于完成后最多保持 10 步被截断，B 必须足够大
    # B = 完成单步奖励；过程单步上限约 2.0*progress + 0.3 + 1.5 ≈ 4.0
    # 400 步过程上限 ≈ 4.0 * 400 = 1600；10*B > 3*1600=4800 => B > 480
    # 取 B = 600 确保主导
    completion_bonus = 0.0
    if inside_dock > 0.0 and heading_ok > 0.0 and speed_ok > 0.0:
        completion_bonus = 600.0

    # ============ 组件 8：动作平滑（轻量，可选）============
    drive = action[0]
    steer = action[1]
    action_smooth = -0.02 * (drive * drive + steer * steer)

    # ---- 汇总 ----
    components = {}
    components["crate_progress"] = float(crate_progress_reward)
    components["crate_near"] = float(crate_near_reward)
    components["docking_quality"] = float(docking_quality)
    components["soft_contact"] = float(soft_contact)
    components["speed_penalty"] = float(speed_penalty)
    components["out_of_bounds"] = float(out_penalty)
    components["completion_bonus"] = float(completion_bonus)
    components["action_smooth"] = float(action_smooth)

    total_reward = (
        crate_progress_reward
        + crate_near_reward
        + docking_quality
        + soft_contact
        + speed_penalty
        + out_penalty
        + completion_bonus
        + action_smooth
    )

    return (float(total_reward), components)
```
