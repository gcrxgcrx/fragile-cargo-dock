```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------------- 几何与恢复 ----------------
    # 货箱到泊位的有符号偏移（归一化：x 除以半宽，y 除以半高）
    dx_now = obs[12]
    dy_now = obs[13]
    dx_nxt = next_obs[12]
    dy_nxt = next_obs[13]

    # 泊位容差（来自环境事实）：|dx| <= 0.024, |dy| <= 0.030
    dock_tol_x = 0.024
    dock_tol_y = 0.030

    # 货箱世界系速度（归一化，乘 3.0 得 m/s）
    cvx_n = next_obs[8]
    cvy_n = next_obs[9]
    crate_speed = ((cvx_n * 3.0) ** 2 + (cvy_n * 3.0) ** 2) ** 0.5

    # 货箱朝向误差（弧度），归一化到 [-pi, pi]
    ch = next_obs[10]
    sh = next_obs[11]
    crate_angle = 0.0
    if ch != 0.0 or sh != 0.0:
        crate_angle = (sh / ((ch * ch + sh * sh) ** 0.5)) if False else 0.0
    # 用 atan2 无法 import math，用近似：通过 cos/sin 直接算误差余弦
    # 泊位朝向假设与 x 轴对齐（货箱需朝向对齐泊位）
    cos_err = ch  # 与目标朝向（cos=1, sin=0）的夹角余弦
    if cos_err > 1.0:
        cos_err = 1.0
    if cos_err < -1.0:
        cos_err = -1.0
    # 朝向误差 < 30° 等价于 cos_err > cos(30°)
    cos_30 = 0.86602540378

    # ---------------- 组件 1：货箱向泊位的增量进展 ----------------
    # 用归一化距离的减少量作为增量信号（避免悬停收割）
    dist_now = (dx_now * dx_now + dy_now * dy_now) ** 0.5
    dist_nxt = (dx_nxt * dx_nxt + dy_nxt * dy_nxt) ** 0.5
    progress = dist_now - dist_nxt  # 正=更接近泊位
    # 限制单步量级，避免异常大跳
    if progress > 0.05:
        progress = 0.05
    if progress < -0.05:
        progress = -0.05
    crate_to_dock_progress = 40.0 * progress

    # ---------------- 组件 2：泊位内位置质量（门控，仅在容差内给分） ----------------
    # 门控：只有货箱中心进入容差区才激活，且用连续衰减避免硬边界
    in_x = max(0.0, 1.0 - abs(dx_nxt) / dock_tol_x)
    in_y = max(0.0, 1.0 - abs(dy_nxt) / dock_tol_y)
    inside_gate = in_x * in_y  # [0,1]，完全进入=1

    # ---------------- 组件 3：朝向对齐质量（门控） ----------------
    if cos_err >= cos_30:
        align_factor = (cos_err - cos_30) / (1.0 - cos_30)
    else:
        align_factor = 0.0
    if align_factor > 1.0:
        align_factor = 1.0

    # ---------------- 组件 4：静止质量（门控，速度 < 0.05 m/s） ----------------
    # 速度阈值 0.05 m/s，用连续衰减
    speed_gate = max(0.0, 1.0 - crate_speed / 0.5)  # 0.5 为过渡尺度
    if speed_gate > 1.0:
        speed_gate = 1.0

    # 联合停靠质量：位置 * 朝向 * 静止（几何平均避免塌缩）
    dock_quality = (inside_gate * align_factor * speed_gate) ** (1.0 / 3.0)
    crate_docking_quality = 15.0 * dock_quality

    # ---------------- 组件 5：完成事件奖励（一次性大额） ----------------
    # 完成条件：完全进入 + 朝向 < 30° + 速度 < 0.05 m/s
    # 用 next_obs 显式推断，容差严格取自环境事实
    fully_inside = (abs(dx_nxt) <= dock_tol_x) and (abs(dy_nxt) <= dock_tol_y)
    aligned = cos_err >= cos_30
    still = crate_speed < 0.05
    completion_event = 0.0
    if fully_inside and aligned and still:
        # 单步完成奖励 B，需满足 10*B > 3*(过程信号上限之和*400)
        # 过程信号单步上限约 40*0.05 + 15 + 少量 = 约 20
        # 3*20*400 = 24000，10*B > 24000 => B > 2400
        completion_event = 3000.0

    # ---------------- 组件 6：轻柔接触（抑制硬碰撞，仅在接触时轻罚速度） ----------------
    # 接触时货箱速度过大 => 可能硬碰撞，用 hinge 轻罚
    contact = next_obs[14]
    soft_contact_penalty = 0.0
    if contact > 0.5:
        # 仅在货箱速度较高时惩罚（避免误罚正常推动）
        excess = crate_speed - 0.3
        if excess > 0.0:
            soft_contact_penalty = -2.0 * excess

    # ---------------- 组件 7：接近泊位时的速度抑制（门控） ----------------
    # 仅在货箱接近泊位（距离 < 0.15 归一化）时激活，抑制高速滑过
    near_dock_gate = max(0.0, 1.0 - dist_nxt / 0.15)
    crate_speed_penalty_near_dock = 0.0
    if near_dock_gate > 0.0:
        excess_spd = crate_speed - 0.05
        if excess_spd > 0.0:
            crate_speed_penalty_near_dock = -3.0 * near_dock_gate * excess_spd

    # ---------------- 组件 8：越界惩罚（hinge，接近边界才罚） ----------------
    # 小车位置 obs[0], obs[1] 归一化到 [-2, 2]，边界在 |x|>0.9 附近
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    out_of_bounds_penalty = 0.0
    if abs(cart_x) > 0.9:
        out_of_bounds_penalty -= 5.0 * (abs(cart_x) - 0.9)
    if abs(cart_y) > 0.9:
        out_of_bounds_penalty -= 5.0 * (abs(cart_y) - 0.9)
    # 货箱越界：货箱到泊位偏移过大（超出仓库范围）
    if abs(dx_nxt) > 1.5:
        out_of_bounds_penalty -= 5.0 * (abs(dx_nxt) - 1.5)
    if abs(dy_nxt) > 1.5:
        out_of_bounds_penalty -= 5.0 * (abs(dy_nxt) - 1.5)

    # ---------------- 组件 9：动作平滑（轻量） ----------------
    action_smoothness = -0.5 * (action[0] * action[0] + action[1] * action[1])

    # ---------------- 汇总 ----------------
    components = {
        "crate_to_dock_progress": crate_to_dock_progress,
        "crate_docking_quality": crate_docking_quality,
        "completion_event": completion_event,
        "soft_contact_penalty": soft_contact_penalty,
        "crate_speed_penalty_near_dock": crate_speed_penalty_near_dock,
        "out_of_bounds_penalty": out_of_bounds_penalty,
        "action_smoothness": action_smoothness,
    }

    total_reward = (
        crate_to_dock_progress
        + crate_docking_quality
        + completion_event
        + soft_contact_penalty
        + crate_speed_penalty_near_dock
        + out_of_bounds_penalty
        + action_smoothness
    )

    return float(total_reward), components
```