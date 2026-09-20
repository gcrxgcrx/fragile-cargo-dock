```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 几何与阈值（来自环境事实） ----------
    # 泊位容差：|obs[12]| <= 0.024, |obs[13]| <= 0.030
    # 朝向误差 < 30 deg -> cos(err) > cos(30) = 0.8660254
    # 速度 < 0.05 m/s -> 归一化速度（/3.0）< 0.0166667
    dock_x_tol = 0.024
    dock_y_tol = 0.030
    cos_align_tol = 0.8660254
    speed_tol = 0.01666667

    # ---------- 货箱到泊位度量（归一化坐标） ----------
    dx_cur = obs[12]
    dy_cur = obs[13]
    dx_nxt = next_obs[12]
    dy_nxt = next_obs[13]

    dist_cur = (dx_cur * dx_cur + dy_cur * dy_cur) ** 0.5
    dist_nxt = (dx_nxt * dx_nxt + dy_nxt * dy_nxt) ** 0.5

    # ---------- 货箱速度（归一化） ----------
    vx_nxt = next_obs[8]
    vy_nxt = next_obs[9]
    crate_speed_nxt = (vx_nxt * vx_nxt + vy_nxt * vy_nxt) ** 0.5

    # ---------- 朝向对齐误差（货箱朝向 vs 泊位朝向=0） ----------
    ch = next_obs[10]
    sh = next_obs[11]
    norm_h = (ch * ch + sh * sh) ** 0.5
    if norm_h > 1e-6:
        cos_err = ch / norm_h
    else:
        cos_err = 0.0
    if cos_err > 1.0:
        cos_err = 1.0
    if cos_err < -1.0:
        cos_err = -1.0

    # =====================================================
    # 组件 1：货箱向泊位的增量进展（主过程信号，增量形式）
    # 只在“这一帧更接近”时给分，避免悬停收割
    # =====================================================
    progress = dist_cur - dist_nxt
    if progress < 0.0:
        progress = 0.0
    if progress > 0.5:
        progress = 0.5
    crate_progress_reward = 6.0 * progress

    # =====================================================
    # 组件 2：接近泊位时的联合对齐/静止引导（门控在“已接近”区域）
    # 只在货箱接近泊位时激活，不阻碍推进
    # =====================================================
    # 接近度因子：dist 越小越大，仅在 dist < 0.20 区间内生效
    near_gate = 1.0 - dist_nxt / 0.20
    if near_gate < 0.0:
        near_gate = 0.0
    if near_gate > 1.0:
        near_gate = 1.0

    # 朝向对齐因子（连续，0..1）
    align_factor = (cos_err - 0.5) / (1.0 - 0.5)
    if align_factor < 0.0:
        align_factor = 0.0
    if align_factor > 1.0:
        align_factor = 1.0

    # 静止因子（连续，0..1）
    slow_factor = 1.0 - crate_speed_nxt / 0.10
    if slow_factor < 0.0:
        slow_factor = 0.0
    if slow_factor > 1.0:
        slow_factor = 1.0

    dock_quality_reward = 1.5 * near_gate * (0.4 * align_factor + 0.6 * slow_factor)

    # =====================================================
    # 组件 3：完成事件（一次性大额奖励，主导过程信号）
    # 条件：货箱完全在泊位内 + 朝向对齐 + 几乎静止
    # =====================================================
    inside_dock = (abs(dx_nxt) <= dock_x_tol) and (abs(dy_nxt) <= dock_y_tol)
    aligned = cos_err >= cos_align_tol
    stopped = crate_speed_nxt < speed_tol

    if inside_dock and aligned and stopped:
        # B = 4000.0，满足 10*B > 3*(过程上限之和 * 400)
        # 过程上限：6.0*0.5 + 1.5*1.0 = 4.5/step；4.5*400*3 = 5400；10*4000=40000 > 5400
        completion_reward = 4000.0
    else:
        completion_reward = 0.0

    # =====================================================
    # 组件 4：软接触惩罚（仅在接触且货箱速度突增时轻罚，避免误伤正常推动）
    # 用接触标志 + 货箱速度突变间接推断，量级很小
    # =====================================================
    contact_now = next_obs[14]
    soft_contact_penalty = 0.0
    if contact_now > 0.5:
        # 货箱速度相对上一帧的突增（世界系归一化速度差）
        vx_cur = obs[8]
        vy_cur = obs[9]
        dvx = vx_nxt - vx_cur
        dvy = vy_nxt - vy_cur
        speed_jump = (dvx * dvx + dvy * dvy) ** 0.5
        # 只在明显速度突变时给轻罚，阈值 0.10（归一化）
        if speed_jump > 0.10:
            soft_contact_penalty = -0.5 * (speed_jump - 0.10)

    # =====================================================
    # 组件 5：越界 hinge 惩罚（小车接近仓库边界时）
    # 小车位置归一化后应在 [-1, 1] 内，留 0.05 余量
    # =====================================================
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    oob_penalty = 0.0
    margin = 0.05
    limit = 1.0 - margin
    if cart_x > limit:
        oob_penalty -= 2.0 * (cart_x - limit)
    if cart_x < -limit:
        oob_penalty -= 2.0 * (-limit - cart_x)
    if cart_y > limit:
        oob_penalty -= 2.0 * (cart_y - limit)
    if cart_y < -limit:
        oob_penalty -= 2.0 * (-limit - cart_y)

    # =====================================================
    # 组件 6：障碍接近度轻罚（前/左/右传感器，防止硬撞墙）
    # 仅在接近度 > 0.6 时生效
    # =====================================================
    obstacle_penalty = 0.0
    sf = next_obs[15]
    sl = next_obs[16]
    sr = next_obs[17]
    if sf > 0.6:
        obstacle_penalty -= 0.5 * (sf - 0.6)
    if sl > 0.6:
        obstacle_penalty -= 0.5 * (sl - 0.6)
    if sr > 0.6:
        obstacle_penalty -= 0.5 * (sr - 0.6)

    # =====================================================
    # 组件 7：动作平滑（轻量，避免抖动，不压制必要推动）
    # =====================================================
    action_smoothness = -0.02 * (action[0] * action[0] + action[1] * action[1])

    # ---------- 汇总 ----------
    total_reward = (
        crate_progress_reward
        + dock_quality_reward
        + completion_reward
        + soft_contact_penalty
        + oob_penalty
        + obstacle_penalty
        + action_smoothness
    )

    components = {
        "crate_progress_reward": float(crate_progress_reward),
        "dock_quality_reward": float(dock_quality_reward),
        "completion_reward": float(completion_reward),
        "soft_contact_penalty": float(soft_contact_penalty),
        "out_of_bounds_penalty": float(oob_penalty),
        "obstacle_penalty": float(obstacle_penalty),
        "action_smoothness": float(action_smoothness),
    }

    return float(total_reward), components
```