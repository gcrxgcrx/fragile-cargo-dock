```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 常量：泊位几何与阈值（来自环境事实） ----
    # 泊位容差：|obs[12]| <= 0.024, |obs[13]| <= 0.030
    dock_tol_x = 0.024
    dock_tol_y = 0.030
    # 货箱速度阈值 0.05 m/s -> 归一化 (除以 3.0)
    speed_thresh_norm = 0.05 / 3.0
    # 朝向误差 < 30 度
    heading_tol = 0.5235987755982988  # 30 deg in rad

    # ---- 货箱到泊位偏移（有符号，已归一化） ----
    off_x = next_obs[12]
    off_y = next_obs[13]
    prev_off_x = obs[12]
    prev_off_y = obs[13]

    # 米制距离（半宽=5.0, 半高=4.0，由卡片推断）
    dist_prev = ((prev_off_x * 5.0) ** 2 + (prev_off_y * 4.0) ** 2) ** 0.5
    dist_now = ((off_x * 5.0) ** 2 + (off_y * 4.0) ** 2) ** 0.5

    # ---- 货箱世界系速度（归一化 -> m/s） ----
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_speed = (crate_vx ** 2 + crate_vy ** 2) ** 0.5

    # ---- 货箱朝向误差 ----
    crate_ang = 0.0
    # atan2 手写近似：用 cos/sin 构造误差角
    cc = next_obs[10]
    cs = next_obs[11]
    # 泊位朝向假定为 0（对齐轴），误差角 = atan2(sin, cos)
    # 用 arcsin 近似小角度不可靠，改用 cos 关系：err ~ acos(clip(cc))
    if cc > 1.0:
        cc = 1.0
    if cc < -1.0:
        cc = -1.0
    # 通过 cos 得到角度大小，符号由 sin 决定（仅用于 shaping 幅度）
    crate_ang = cc

    # ---- 组件 1：货箱向泊位推进的增量信号 ----
    # 只在"这一帧更接近"时给分；远离时给轻微惩罚
    progress_delta = dist_prev - dist_now
    crate_to_dock_progress = 2.0 * progress_delta

    # ---- 组件 2：进入泊位的联合质量信号（门控式，不全局收分） ----
    # 位置因子：越接近容差越好
    pos_err_x = abs(off_x) / dock_tol_x
    pos_err_y = abs(off_y) / dock_tol_y
    if pos_err_x > 3.0:
        pos_err_x = 3.0
    if pos_err_y > 3.0:
        pos_err_y = 3.0
    f_pos = max(0.0, 1.0 - 0.5 * (pos_err_x + pos_err_y))
    # 朝向因子
    f_head = max(0.0, cc)
    # 速度因子：只在接近泊位时启用
    near_dock = max(0.0, 1.0 - dist_now / 1.5)
    f_speed = 1.0 / (1.0 + 20.0 * crate_speed)
    # 联合质量（几何平均避免塌缩）
    quality = (f_pos * f_head * f_speed) ** (1.0 / 3.0)
    crate_docking_quality = 1.0 * quality * near_dock

    # ---- 组件 3：完成事件（一次性大额奖励） ----
    # 判定：货箱完全进入泊位 + 朝向对齐 + 静止
    inside = (abs(off_x) <= dock_tol_x) and (abs(off_y) <= dock_tol_y)
    aligned = (cc >= 0.8660254037844387)  # cos(30deg)
    still = (crate_speed < 0.05)
    if inside and aligned and still:
        completion_bonus = 5000.0
    else:
        completion_bonus = 0.0

    # ---- 组件 4：接近泊位时的货箱速度抑制（门控，仅接近时激活） ----
    if near_dock > 0.0 and crate_speed > speed_thresh_norm:
        crate_speed_penalty_near_dock = -0.5 * near_dock * (crate_speed - speed_thresh_norm)
    else:
        crate_speed_penalty_near_dock = 0.0

    # ---- 组件 5：接触时的软接触惩罚（间接推断，轻量） ----
    contact = next_obs[14]
    cart_fwd = next_obs[4] * 3.0
    if contact > 0.5 and crate_speed > 1.0:
        soft_contact_penalty = -0.2 * (crate_speed - 1.0)
    else:
        soft_contact_penalty = 0.0

    # ---- 组件 6：越界惩罚（hinge，仅边界附近） ----
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    out_of_bounds_penalty = 0.0
    lim = 0.9
    if abs(cart_x) > lim:
        out_of_bounds_penalty -= 1.0 * (abs(cart_x) - lim)
    if abs(cart_y) > lim:
        out_of_bounds_penalty -= 1.0 * (abs(cart_y) - lim)

    # ---- 组件 7：动作平滑（轻量） ----
    action_smoothness = -0.02 * (action[0] ** 2 + action[1] ** 2)

    total_reward = (
        crate_to_dock_progress
        + crate_docking_quality
        + completion_bonus
        + crate_speed_penalty_near_dock
        + soft_contact_penalty
        + out_of_bounds_penalty
        + action_smoothness
    )

    components = {
        "crate_to_dock_progress": crate_to_dock_progress,
        "crate_docking_quality": crate_docking_quality,
        "completion_bonus": completion_bonus,
        "crate_speed_penalty_near_dock": crate_speed_penalty_near_dock,
        "soft_contact_penalty": soft_contact_penalty,
        "out_of_bounds_penalty": out_of_bounds_penalty,
        "action_smoothness": action_smoothness,
    }

    return float(total_reward), components
```