```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 常量（来自环境事实）----
    HALF_W = 5.0      # 仓库半宽（米），obs[0]*HALF_W = cart_x
    HALF_H = 4.0      # 仓库半高（米）
    CRATE_SCALE = 3.0 # 货箱相对位置/速度尺度

    # 泊位容差（来自环境事实：|obs[12]|<=0.024, |obs[13]|<=0.030）
    DOCK_TOL_X = 0.024
    DOCK_TOL_Y = 0.030
    DOCK_SPEED_TOL = 0.05 / 3.0   # obs[8]/obs[9] 单位化后阈值
    ANGLE_TOL = 30.0 * 3.141592653589793 / 180.0

    # ---- 货箱到泊位距离（用 obs[12],obs[13] 恢复米制）----
    dx_m = obs[12] * HALF_W
    dy_m = obs[13] * HALF_H
    dist = (dx_m * dx_m + dy_m * dy_m) ** 0.5

    ndx_m = next_obs[12] * HALF_W
    ndy_m = next_obs[13] * HALF_H
    next_dist = (ndx_m * ndx_m + ndy_m * ndy_m) ** 0.5

    # ---- 货箱速度（单位化后）----
    vx = obs[8]
    vy = obs[9]
    speed = (vx * vx + vy * vy) ** 0.5
    next_speed = (next_obs[8] * next_obs[8] + next_obs[9] * next_obs[9]) ** 0.5

    # ---- 朝向误差 ----
    # 货箱朝向：atan2(obs[11], obs[10])；泊位朝向对齐目标（假定泊位朝向为 0 或与货箱初始一致）
    # 用货箱朝向与 0 的偏差作为对齐度代理（环境事实：朝向误差 < 30°）
    heading = 0.0
    if obs[10] != 0.0 or obs[11] != 0.0:
        # 用 cos 分量近似角度误差（对齐时 cos≈1）
        heading = obs[10]
    angle_err = 1.0 - heading  # 0 表示完全对齐

    # ---- 是否在泊位容差内 ----
    in_dock_pos = 1.0 if (abs(obs[12]) <= DOCK_TOL_X and abs(obs[13]) <= DOCK_TOL_Y) else 0.0
    in_dock_pos_next = 1.0 if (abs(next_obs[12]) <= DOCK_TOL_X and abs(next_obs[13]) <= DOCK_TOL_Y) else 0.0

    # ---- 组件 1：货箱向泊位推进（增量形式，避免悬停陷阱）----
    progress = dist - next_dist  # 正值表示这一帧更接近
    crate_progress_reward = 3.0 * progress

    # ---- 组件 2：接近度门控的精细塑形（仅在接近泊位时激活）----
    # 用平滑门控：距离越近门控越大，但只在 < 0.5m 范围
    near_gate = max(0.0, 1.0 - dist / 0.5)
    # 位置精细度：在泊位内时给正信号
    pos_fine = 0.0
    if in_dock_pos > 0.0:
        pos_fine = 1.0
    # 朝向对齐（接近时激活）
    align_factor = max(0.0, 1.0 - angle_err / 0.5)
    dock_quality_reward = 0.5 * near_gate * (pos_fine + align_factor)

    # ---- 组件 3：接近泊位时的速度抑制（门控，仅在接近时激活）----
    # 只在 dist < 0.3m 时启用，避免阻碍到达
    speed_gate = max(0.0, 1.0 - dist / 0.3)
    speed_penalty = -0.3 * speed_gate * speed

    # ---- 组件 4：软接触惩罚（间接推断，轻量）----
    # 接触时速度突变大 → 惩罚；但正常推动也应允许
    contact = obs[14]
    contact_penalty = 0.0
    if contact > 0.5:
        # 货箱速度突变作为硬碰撞代理
        speed_change = abs(next_speed - speed)
        if speed_change > 0.15:
            contact_penalty = -0.5 * (speed_change - 0.15)

    # ---- 组件 5：越界惩罚（hinge 形式）----
    # 小车位置边界：obs[0],obs[1] 在 [-1,1] 内安全
    cart_x = obs[0]
    cart_y = obs[1]
    oob_penalty = 0.0
    if abs(cart_x) > 0.9:
        oob_penalty -= 1.0 * (abs(cart_x) - 0.9)
    if abs(cart_y) > 0.9:
        oob_penalty -= 1.0 * (abs(cart_y) - 0.9)
    # 货箱越界：obs[12],obs[13] 恢复后判断
    crate_x_world = obs[12] * HALF_W
    crate_y_world = obs[13] * HALF_H
    if abs(crate_x_world) > 4.5:
        oob_penalty -= 1.0 * (abs(crate_x_world) - 4.5)
    if abs(crate_y_world) > 3.5:
        oob_penalty -= 1.0 * (abs(crate_y_world) - 3.5)

    # ---- 组件 6：动作平滑（轻量）----
    smooth_penalty = -0.02 * (action[0] * action[0] + action[1] * action[1])

    # ---- 组件 7：完成事件（大额奖励，主导过程信号）----
    # 完成条件：货箱在泊位容差内 + 朝向对齐 + 速度 < 0.05 m/s
    # 由于 episode 在完成后立即结束，这里给一次性大额奖励
    # 量级判据：B > 3 * (过程信号上限之和 * 400) / 10
    # 过程信号单步上限估算：progress 最大约 0.01*3=0.03, quality 0.5*2=1.0, 其他惩罚...
    # 保守取过程信号单步上限之和约 1.5，则 B > 3 * 1.5 * 400 / 10 = 180
    # 取 B = 500 确保主导
    docked_success = 0.0
    if in_dock_pos > 0.0 and next_speed < DOCK_SPEED_TOL and angle_err < 0.5:
        docked_success = 500.0

    # ---- 总奖励 ----
    total = (
        crate_progress_reward
        + dock_quality_reward
        + speed_penalty
        + contact_penalty
        + oob_penalty
        + smooth_penalty
        + docked_success
    )

    components = {
        "crate_progress": float(crate_progress_reward),
        "dock_quality": float(dock_quality_reward),
        "speed_penalty": float(speed_penalty),
        "contact_penalty": float(contact_penalty),
        "oob_penalty": float(oob_penalty),
        "smooth_penalty": float(smooth_penalty),
        "docked_success": float(docked_success),
    }

    return float(total), components
```