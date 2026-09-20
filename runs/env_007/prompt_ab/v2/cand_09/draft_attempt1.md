```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 几何/阈值常量（来自环境事实）----
    # 泊位容差：|obs[12]| <= 0.024, |obs[13]| <= 0.030
    dock_x_tol = 0.024
    dock_y_tol = 0.030
    # 朝向误差 < 30 度
    angle_tol = 0.5235987755982988  # 30 deg in rad
    # 速度 < 0.05 m/s -> 归一化后 = 0.05 / 3.0
    speed_tol_norm = 0.05 / 3.0

    # ---- 当前与下一步的泊位偏移（归一化量，直接可用）----
    dx_now = obs[12]
    dy_now = obs[13]
    dx_next = next_obs[12]
    dy_next = next_obs[13]

    # 归一化距离度量（用容差归一，使接近程度与容差同尺度）
    d_now = ((dx_now / dock_x_tol) ** 2 + (dy_now / dock_y_tol) ** 2) ** 0.5
    d_next = ((dx_next / dock_x_tol) ** 2 + (dy_next / dock_y_tol) ** 2) ** 0.5

    # ---- 货箱速度（世界系，归一化）----
    vx_next = next_obs[8] * 3.0
    vy_next = next_obs[9] * 3.0
    crate_speed = (vx_next ** 2 + vy_next ** 2) ** 0.5

    # ---- 货箱朝向误差 ----
    crate_ang = 0.0
    if (next_obs[10] != 0.0) or (next_obs[11] != 0.0):
        # atan2 近似：用比例判断象限，避免 import math
        # 用简单方式：朝向误差用 |sin| 与 |cos| 组合衡量
        crate_cos = next_obs[10]
        crate_sin = next_obs[11]
        # 与目标朝向（假设泊位朝向为 +x，cos=1,sin=0）的对齐误差
        # 1 - cos 越小越对齐；同时用 |sin| 捕捉 180 度歧义
        crate_ang = (crate_sin ** 2 + (1.0 - crate_cos) ** 2) ** 0.5
    else:
        crate_ang = 1.0

    # =========================================================
    # 组件 1：货箱向泊位推进（增量形式，避免悬停收割）
    # 只有"这一帧更接近泊位"才给正分；远离给负分
    # =========================================================
    progress_delta = d_now - d_next  # 正 = 更接近
    # 压缩到有界范围，避免极端值
    progress_raw = progress_delta / (1.0 + abs(progress_delta))
    crate_to_dock_progress = 1.0 * progress_raw

    # =========================================================
    # 组件 2：接近泊位时的停靠质量门控（仅在接近时激活）
    # 联合条件：位置接近 + 朝向对齐 + 速度低
    # 用连续 bounded factor，几何平均避免塌缩
    # =========================================================
    # 位置接近因子：d_next 越小越接近 1
    pos_factor = 1.0 / (1.0 + d_next)
    # 朝向对齐因子
    ang_factor = 1.0 / (1.0 + crate_ang / angle_tol)
    # 速度因子
    spd_factor = 1.0 / (1.0 + crate_speed / 0.05)
    # 仅在接近泊位时激活（d_next < 3 倍容差）
    near_gate = 0.0
    if d_next < 3.0:
        near_gate = (3.0 - d_next) / 3.0
        if near_gate < 0.0:
            near_gate = 0.0
        if near_gate > 1.0:
            near_gate = 1.0
    # 几何平均（三个因子）
    dock_quality = (pos_factor * ang_factor * spd_factor) ** (1.0 / 3.0)
    crate_docking_quality = 0.5 * near_gate * dock_quality

    # =========================================================
    # 组件 3：接近泊位时的速度抑制（hinge，仅在接近时启用）
    # 目的：避免货箱高速滑过泊位
    # =========================================================
    crate_speed_penalty_near_dock = 0.0
    if d_next < 3.0:
        # 速度超过 0.05 m/s 的部分给惩罚（归一化后）
        excess = crate_speed - 0.05
        if excess > 0.0:
            # 只在接近泊位时惩罚，权重随接近程度增大
            proximity_weight = (3.0 - d_next) / 3.0
            crate_speed_penalty_near_dock = -0.3 * proximity_weight * (excess / 0.05)

    # =========================================================
    # 组件 4：软接触惩罚（间接推断：接触 + 速度突变）
    # 不可靠，权重小，避免误判正常推动
    # =========================================================
    contact_now = obs[14]
    contact_next = next_obs[14]
    soft_contact_penalty = 0.0
    if (contact_next > 0.5) and (contact_now < 0.5):
        # 刚发生接触，若货箱速度较大则可能为硬碰撞
        if crate_speed > 0.5:
            soft_contact_penalty = -0.1 * (crate_speed / 3.0)

    # =========================================================
    # 组件 5：越界惩罚（hinge，仅在接近边界时生效）
    # 小车位置 obs[0], obs[1] 归一化到 [-2,2] 裁剪
    # 货箱相对小车位置 obs[6], obs[7] 可用于推断货箱世界位置
    # =========================================================
    out_of_bounds_penalty = 0.0
    # 小车边界：|obs[0]|, |obs[1]| 接近 1 时预警
    cart_x = obs[0]
    cart_y = obs[1]
    if abs(cart_x) > 0.8:
        out_of_bounds_penalty -= 0.2 * (abs(cart_x) - 0.8)
    if abs(cart_y) > 0.8:
        out_of_bounds_penalty -= 0.2 * (abs(cart_y) - 0.8)
    # 货箱到泊位偏移的边界预警：若偏移过大说明货箱远离泊位
    if abs(dx_now) > 1.5:
        out_of_bounds_penalty -= 0.1 * (abs(dx_now) - 1.5)
    if abs(dy_now) > 1.5:
        out_of_bounds_penalty -= 0.1 * (abs(dy_now) - 1.5)

    # =========================================================
    # 组件 6：完成事件奖励（一次性大额）
    # 完成条件：|obs[12]|<=0.024, |obs[13]|<=0.030,
    #           朝向误差 < 30deg, 速度 < 0.05 m/s
    # 由于完成后 episode 立即结束（最多再累积保持步数），
    # 单步完成奖励 B 需满足 10*B > 3*(过程组件单步上限之和 * 400)
    # 过程组件单步上限估计：progress(1.0) + dock_quality(0.5) + 
    #                        speed_penalty(0) + contact(0) + oob(0) ≈ 1.5
    # 3 * 1.5 * 400 = 1800，10*B > 1800 -> B > 180
    # 取 B = 500 确保主导
    # =========================================================
    docked_bonus = 0.0
    inside_x = abs(dx_next) <= dock_x_tol
    inside_y = abs(dy_next) <= dock_y_tol
    aligned = crate_ang < angle_tol
    slow = crate_speed < 0.05
    if inside_x and inside_y and aligned and slow:
        docked_bonus = 500.0

    # =========================================================
    # 总奖励
    # =========================================================
    total_reward = (
        crate_to_dock_progress
        + crate_docking_quality
        + crate_speed_penalty_near_dock
        + soft_contact_penalty
        + out_of_bounds_penalty
        + docked_bonus
    )

    components = {
        "crate_to_dock_progress": crate_to_dock_progress,
        "crate_docking_quality": crate_docking_quality,
        "crate_speed_penalty_near_dock": crate_speed_penalty_near_dock,
        "soft_contact_penalty": soft_contact_penalty,
        "out_of_bounds_penalty": out_of_bounds_penalty,
        "docked_bonus": docked_bonus,
    }

    return (float(total_reward), components)
```