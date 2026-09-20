# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------------- 几何与阈值（来自环境事实） ----------------
    # 泊位容差：|obs[12]| <= 0.024, |obs[13]| <= 0.030
    # 朝向误差 < 30deg, 速度 < 0.05 m/s, 连续保持 10 步
    pos_tol_x = 0.024
    pos_tol_y = 0.030
    speed_tol = 0.05
    angle_tol_cos = 0.8660254037844387  # cos(30deg)

    # 半宽/半高（用于把归一化偏移还原成米，仅用于 shaping 尺度）
    half_w = 5.0
    half_h = 4.0

    # ---------------- 货箱到泊位距离（前后两帧） ----------------
    dx_old = obs[12] * half_w
    dy_old = obs[13] * half_h
    dx_new = next_obs[12] * half_w
    dy_new = next_obs[13] * half_h

    dist_old = (dx_old * dx_old + dy_old * dy_old) ** 0.5
    dist_new = (dx_new * dx_new + dy_new * dy_new) ** 0.5

    # ---------------- 货箱速度 ----------------
    cvx = next_obs[8] * 3.0
    cvy = next_obs[9] * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---------------- 货箱朝向误差 ----------------
    crate_cos = next_obs[10]
    crate_sin = next_obs[11]
    heading_norm = (crate_cos * crate_cos + crate_sin * crate_sin) ** 0.5
    if heading_norm < 1.0e-6:
        heading_norm = 1.0e-6
    align_cos = crate_cos / heading_norm

    # ---------------- 主信号 1：货箱到泊位增量进展 ----------------
    # 只在"这一帧更接近泊位"时给分，避免悬停收割
    progress = dist_old - dist_new
    if progress < 0.0:
        progress = 0.0
    r_progress = 6.0 * progress

    # ---------------- 主信号 2：接近泊位时的对接质量（门控 + 联合条件） ----------------
    # 位置门：距离越近越接近 1
    gate_pos = 1.0 - dist_new / 1.0
    if gate_pos < 0.0:
        gate_pos = 0.0
    if gate_pos > 1.0:
        gate_pos = 1.0

    # 朝向因子：cos 误差映射到 [0,1]
    if align_cos < 0.0:
        align_cos = 0.0
    f_align = align_cos * align_cos

    # 速度因子：越慢越接近 1
    f_speed = 1.0 / (1.0 + 4.0 * crate_speed)

    # 联合对接质量（几何平均，避免乘积塌缩）
    dock_quality = (gate_pos * f_align * f_speed) ** (1.0 / 3.0)
    r_dock_quality = 1.5 * dock_quality

    # ---------------- 主信号 3：完成事件（大额一次性奖励） ----------------
    # 显式从 obs 推断完成条件
    inside = 0.0
    if abs(next_obs[12]) <= pos_tol_x and abs(next_obs[13]) <= pos_tol_y:
        inside = 1.0
    aligned = 0.0
    if align_cos >= angle_tol_cos:
        aligned = 1.0
    still = 0.0
    if crate_speed < speed_tol:
        still = 1.0

    completed = inside * aligned * still
    # 完成单步奖励 B，需压过 400 步过程信号总和
    # 过程信号单步上限约 6.0*max_progress + 1.5 ≈ 量级 < 10
    # B 取 4000，远大于 3*(10*400)=12000 的近似上限
    r_complete = 4000.0 * completed

    # ---------------- 辅助约束：轻柔接触 ----------------
    # 仅在接触时，用货箱速度突变间接推断硬碰撞风险
    contact = next_obs[14]
    contact_pen = 0.0
    if contact > 0.5:
        # 接触时货箱速度过大 -> 可能硬碰撞
        excess = crate_speed - 0.5
        if excess > 0.0:
            contact_pen = -0.02 * (excess * excess)
    r_contact = contact_pen

    # ---------------- 辅助约束：接近泊位时抑制货箱速度 ----------------
    # 只在接近泊位（gate_pos 较大）时启用，不阻碍推进
    speed_pen = 0.0
    if gate_pos > 0.3:
        excess_speed = crate_speed - speed_tol
        if excess_speed > 0.0:
            speed_pen = -0.05 * gate_pos * (excess_speed * excess_speed)
    r_speed_near = speed_pen

    # ---------------- 辅助约束：越界 hinge 惩罚 ----------------
    # 小车位置归一化到 [-1,1] 附近，接近边界时惩罚
    oob_pen = 0.0
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    for c in (cart_x, cart_y):
        excess = abs(c) - 0.95
        if excess > 0.0:
            oob_pen = oob_pen - 0.5 * (excess * excess)
    # 货箱到泊位偏移过大也提示越界风险（货箱离开仓库）
    crate_excess = dist_new - 3.0
    if crate_excess > 0.0:
        oob_pen = oob_pen - 0.05 * (crate_excess * crate_excess)
    r_oob = oob_pen

    # ---------------- 辅助约束：动作平滑（轻量，不压制推动） ----------------
    smooth_pen = -0.005 * (action[0] * action[0] + action[1] * action[1])
    r_smooth = smooth_pen

    # ---------------- 汇总 ----------------
    total_reward = (
        r_progress
        + r_dock_quality
        + r_complete
        + r_contact
        + r_speed_near
        + r_oob
        + r_smooth
    )

    components = {
        "crate_to_dock_progress": r_progress,
        "crate_docking_quality": r_dock_quality,
        "dock_complete_event": r_complete,
        "soft_contact_penalty": r_contact,
        "crate_speed_near_dock": r_speed_near,
        "out_of_bounds_penalty": r_oob,
        "action_smoothness": r_smooth,
    }

    return (float(total_reward), components)
```
