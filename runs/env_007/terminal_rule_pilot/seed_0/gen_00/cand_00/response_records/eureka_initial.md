# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 几何/阈值常量（来自环境事实）----
    # 泊位容差：|obs[12]| <= 0.024, |obs[13]| <= 0.030
    # 朝向误差 < 30 deg，速度 < 0.05 m/s，连续保持 10 步
    # 位置偏移为归一化量，恢复米制：x 乘半宽 5.0，y 乘半高 4.0
    HALF_W = 5.0
    HALF_H = 4.0

    # ---- 货箱到泊位偏移（米）----
    dx = next_obs[12] * HALF_W
    dy = next_obs[13] * HALF_H
    dist = (dx * dx + dy * dy) ** 0.5

    dx_p = obs[12] * HALF_W
    dy_p = obs[13] * HALF_H
    dist_prev = (dx_p * dx_p + dy_p * dy_p) ** 0.5

    # ---- 货箱速度（m/s）----
    cvx = next_obs[8] * 3.0
    cvy = next_obs[9] * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---- 货箱朝向误差（rad）----
    crate_ang = 0.0
    if next_obs[10] != 0.0 or next_obs[11] != 0.0:
        crate_ang = (next_obs[11] ** 2 + next_obs[10] ** 2) ** 0.5
    # 用 cos 求角度：err = acos(cos_theta) 近似，避免 atan2
    cos_th = next_obs[10]
    if cos_th > 1.0:
        cos_th = 1.0
    if cos_th < -1.0:
        cos_th = -1.0
    # 朝向误差近似：1 - cos 在 30deg 时约 0.134
    align_err = 1.0 - cos_th

    # ---- 完成判据（显式从 obs 推断）----
    inside = (abs(next_obs[12]) <= 0.024) and (abs(next_obs[13]) <= 0.030)
    aligned = align_err < 0.134  # cos(30deg) ≈ 0.866 -> 1-cos ≈ 0.134
    slow = crate_speed < 0.05
    docked_now = inside and aligned and slow

    # 连续保持计数（用 obs 传递，不依赖 info）
    # 使用 obs[18] 时间比例无法计数，用 next_obs 与 obs 的 docked 状态做近似：
    # 若当前和上一帧都满足 docked，则视为保持中
    inside_p = (abs(obs[12]) <= 0.024) and (abs(obs[13]) <= 0.030)
    cos_th_p = obs[10]
    if cos_th_p > 1.0:
        cos_th_p = 1.0
    if cos_th_p < -1.0:
        cos_th_p = -1.0
    aligned_p = (1.0 - cos_th_p) < 0.134
    cvx_p = obs[8] * 3.0
    cvy_p = obs[9] * 3.0
    speed_p = (cvx_p * cvx_p + cvy_p * cvy_p) ** 0.5
    docked_prev = inside_p and aligned_p and (speed_p < 0.05)

    # ---- 组件 1：货箱向泊位推进（增量形式，避免悬停收割）----
    progress = dist_prev - dist
    if progress > 0.0:
        crate_progress = 2.0 * progress
    else:
        crate_progress = 0.5 * progress  # 远离时轻罚，不压制必要机动

    # ---- 组件 2：进入泊位容差区的软门控对齐/静止 shaping（仅在接近时激活）----
    # 距离门：越近越强，但只在 < 1.0 m 内激活，避免全局收分
    if dist < 1.0:
        near_gate = 1.0 - dist
        if near_gate < 0.0:
            near_gate = 0.0
    else:
        near_gate = 0.0

    # 朝向对齐因子（只在接近时给）
    align_factor = 1.0 - align_err
    if align_factor < 0.0:
        align_factor = 0.0
    align_term = 0.5 * near_gate * align_factor

    # 静止因子（只在接近时给，且速度越小越高）
    if crate_speed < 0.3:
        speed_factor = 1.0 - crate_speed / 0.3
    else:
        speed_factor = 0.0
    speed_term = 0.5 * near_gate * speed_factor

    # ---- 组件 3：完成事件（一次性大额，主导过程信号）----
    # 完成条件：完全进入 + 朝向对齐 + 静止，且上一帧也满足（保持中）
    if docked_now and docked_prev:
        dock_bonus = 500.0
    elif docked_now:
        dock_bonus = 200.0  # 首次满足，给中等奖励
    else:
        dock_bonus = 0.0

    # ---- 组件 4：轻柔接触约束（仅在接触时，且速度突变大时惩罚）----
    contact = next_obs[14]
    if contact > 0.5:
        # 接触时，若货箱速度高，视为硬推，轻罚
        if crate_speed > 0.5:
            contact_pen = -0.3 * (crate_speed - 0.5)
        else:
            contact_pen = 0.0
    else:
        contact_pen = 0.0

    # ---- 组件 5：越界防护（hinge 形式，仅在接近边界时）----
    # 小车位置 obs[0], obs[1] 归一化到 [-2,2]，边界惩罚在 |x|>1.8 或 |y|>1.8
    oob_pen = 0.0
    if abs(next_obs[0]) > 1.8:
        oob_pen -= 0.5 * (abs(next_obs[0]) - 1.8)
    if abs(next_obs[1]) > 1.8:
        oob_pen -= 0.5 * (abs(next_obs[1]) - 1.8)

    # ---- 组件 6：货箱接近泊位时抑制速度（避免滑过）----
    # 仅在距离 < 0.5 m 且速度 > 0.1 m/s 时惩罚
    if dist < 0.5 and crate_speed > 0.1:
        overspeed_pen = -0.4 * (crate_speed - 0.1)
    else:
        overspeed_pen = 0.0

    # ---- 组件 7：动作平滑（轻量，不压制推动）----
    smooth_pen = -0.01 * (action[0] ** 2 + action[1] ** 2)

    # ---- 汇总 ----
    components = {}
    components['crate_progress'] = float(crate_progress)
    components['align_term'] = float(align_term)
    components['speed_term'] = float(speed_term)
    components['dock_bonus'] = float(dock_bonus)
    components['contact_pen'] = float(contact_pen)
    components['oob_pen'] = float(oob_pen)
    components['overspeed_pen'] = float(overspeed_pen)
    components['smooth_pen'] = float(smooth_pen)

    total_reward = (
        crate_progress
        + align_term
        + speed_term
        + dock_bonus
        + contact_pen
        + oob_pen
        + overspeed_pen
        + smooth_pen
    )

    return float(total_reward), components
```
