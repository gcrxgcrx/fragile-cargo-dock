# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 泊位几何与阈值（来自环境事实）----
    # |obs[12]| <= 0.024 且 |obs[13]| <= 0.030 表示货箱完全在泊位内
    # 朝向误差 < 30deg (~0.5236 rad)，速度 < 0.05 m/s，连续保持 10 步
    dock_x_tol = 0.024
    dock_y_tol = 0.030
    heading_tol = 0.5235987755982988   # 30 deg
    speed_tol = 0.05

    # ---- 从 obs 恢复信号 ----
    # 货箱到泊位的有符号偏移（归一化）
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    # 货箱世界系速度（归一化 -> m/s）
    cvx = obs[8] * 3.0
    cvy = obs[9] * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向误差
    crate_ang = (obs[11] * obs[11] + obs[10] * obs[10]) ** 0.5
    if crate_ang < 1.0e-6:
        crate_ang = 1.0e-6
    cos_err = obs[10] / crate_ang
    if cos_err > 1.0:
        cos_err = 1.0
    if cos_err < -1.0:
        cos_err = -1.0
    # 朝向误差角（rad），通过 1-cos 近似单调映射，误差越小值越大
    heading_align = (1.0 + cos_err) * 0.5   # 0..1, 1 为完全对齐

    # 归一化距离度量（用于增量式进度）
    dist_old = (dx * dx + dy * dy) ** 0.5
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    dist_delta = dist_old - dist_new   # 正=更接近

    # 是否在泊位容差区内（连续化）
    in_dock = 1.0
    if abs(dx) > dock_x_tol:
        in_dock = 0.0
    if abs(dy) > dock_y_tol:
        in_dock = 0.0

    # 速度因子：越慢越高，但只在货箱接近泊位时用于门控
    speed_factor = 1.0 / (1.0 + 4.0 * crate_speed)

    # 接近泊位程度（用于门控，不用作全局持续奖励）
    near_dock = 1.0 / (1.0 + 20.0 * dist_old)

    # 接触标志
    contact = obs[14]

    components = {}

    # ---- 主组件 1：增量式货箱到泊位进度（只在更接近时给分）----
    progress_reward = 0.0
    if dist_delta > 0.0:
        progress_reward = 4.0 * dist_delta
    components["crate_to_dock_progress"] = progress_reward

    # ---- 主组件 2：货箱进入泊位容差区的稀疏奖励（小量过程信号，不构成悬停收益）----
    # 仅当货箱真正在容差区内才给，且用连续门控避免跳变
    if in_dock > 0.0:
        components["crate_in_dock_bonus"] = 0.5
    else:
        components["crate_in_dock_bonus"] = 0.0

    # ---- 主组件 3：停靠质量联合门控（位置*朝向*速度），仅在接近泊位时激活 ----
    dock_quality = 0.0
    if near_dock > 0.15:
        pos_factor = 1.0
        if abs(dx) > dock_x_tol:
            pos_factor = 0.0
        if abs(dy) > dock_y_tol:
            pos_factor = 0.0
        if pos_factor > 0.0:
            dock_quality = 1.0 * pos_factor * heading_align * speed_factor
    components["crate_docking_quality"] = dock_quality

    # ---- 完成事件奖励：显式从 obs 推断完成条件，大额奖励主导 ----
    # 完成条件：货箱完全在泊位内 + 朝向误差 < 30deg + 速度 < 0.05 m/s
    success_cond = 0.0
    if in_dock > 0.0:
        if cos_err >= 0.8660254037844387:   # cos(30deg)
            if crate_speed < speed_tol:
                success_cond = 1.0
    if success_cond > 0.0:
        components["docking_complete_event"] = 5000.0
    else:
        components["docking_complete_event"] = 0.0

    # ---- 软接触惩罚：接触且货箱速度较大时轻罚（间接推断硬碰撞风险）----
    soft_penalty = 0.0
    if contact > 0.5:
        if crate_speed > 0.5:
            soft_penalty = -0.3 * (crate_speed - 0.5)
    components["soft_contact_penalty"] = soft_penalty

    # ---- 接近泊位时的速度抑制（仅当接近时才启用，避免阻碍到达）----
    speed_near_dock_penalty = 0.0
    if near_dock > 0.4:
        if crate_speed > speed_tol:
            speed_near_dock_penalty = -0.5 * (crate_speed - speed_tol)
    components["crate_speed_penalty_near_dock"] = speed_near_dock_penalty

    # ---- 越界惩罚：小车或货箱接近仓库边界（hinge）----
    # 小车位置归一化到 [-2,2] 区间，边界取 |obs[0]|>=1 或 |obs[1]|>=1 视为越界
    oob_penalty = 0.0
    cart_x = obs[0]
    cart_y = obs[1]
    if abs(cart_x) > 0.9:
        oob_penalty = oob_penalty - 1.0 * (abs(cart_x) - 0.9)
    if abs(cart_y) > 0.9:
        oob_penalty = oob_penalty - 1.0 * (abs(cart_y) - 0.9)
    # 货箱到泊位的偏移过大也提示货箱可能越界，用较大偏移做弱提示
    if dist_old > 1.5:
        oob_penalty = oob_penalty - 0.5 * (dist_old - 1.5)
    components["out_of_bounds_penalty"] = oob_penalty

    # ---- 动作平滑（轻量，不压制必要推动）----
    smooth_penalty = -0.01 * (action[0] * action[0] + action[1] * action[1])
    components["action_smoothness"] = smooth_penalty

    total_reward = 0.0
    for key in components:
        total_reward = total_reward + components[key]

    return (float(total_reward), components)
```
