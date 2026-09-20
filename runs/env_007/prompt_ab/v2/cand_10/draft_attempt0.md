```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 几何常量（来自环境事实） ----------
    # obs[12] = crate_to_dock_x / 仓库半宽 ; obs[13] = crate_to_dock_y / 仓库半高
    # 成功判据: |obs[12]| <= 0.024 且 |obs[13]| <= 0.030
    DOCK_X_TOL = 0.024
    DOCK_Y_TOL = 0.030
    # 货箱速度: obs[8]*3.0, obs[9]*3.0 (m/s); 成功阈值 0.05 m/s -> 归一化 0.05/3.0
    SPEED_TOL = 0.05 / 3.0
    # 朝向误差 < 30 deg
    ANGLE_TOL = 0.5235987756

    # ---------- 货箱到泊位距离（归一化尺度） ----------
    dx = next_obs[12]
    dy = next_obs[13]
    dist_n = (dx * dx + dy * dy) ** 0.5

    dx0 = obs[12]
    dy0 = obs[13]
    dist_n0 = (dx0 * dx0 + dy0 * dy0) ** 0.5

    # ---------- 组件 1: 货箱向泊位推进（增量形式，避免悬停收割） ----------
    progress = dist_n0 - dist_n
    crate_progress = 6.0 * progress

    # ---------- 组件 2: 接近门控的精细停靠质量（仅在接近泊位时激活） ----------
    # 接近度因子: 距离 <= 0.20 时满值, 到 0.60 衰减到 0
    near = max(0.0, min(1.0, (0.60 - dist_n) / 0.40))

    # 位置因子：越接近容差中心越高
    pos_err = ((dx / DOCK_X_TOL) ** 2 + (dy / DOCK_Y_TOL) ** 2) ** 0.5
    pos_factor = 1.0 / (1.0 + pos_err)

    # 朝向因子：货箱朝向与泊位对齐（货箱朝向误差）
    c_ang = next_obs[10]
    s_ang = next_obs[11]
    ang_err = (s_ang * s_ang + c_ang * c_ang) ** 0.5
    if ang_err < 1e-6:
        ang_err = 1e-6
    # cos 误差近似: 用 |sin| 作为朝向偏差度量
    # 货箱朝向角 theta = atan2(s,c); 偏差对 0 参考 -> 用 |s| 作为对齐偏差
    align_err = abs(s_ang)
    align_factor = max(0.0, 1.0 - align_err / 0.5)

    # 速度因子：货箱速度越小越高
    spd = ((next_obs[8] * next_obs[8]) + (next_obs[9] * next_obs[9])) ** 0.5
    spd_factor = max(0.0, 1.0 - spd / (SPEED_TOL * 4.0))

    dock_quality = 2.0 * near * pos_factor * align_factor * spd_factor

    # ---------- 组件 3: 完成事件（一次性大额奖励） ----------
    inside = 1.0 if (abs(dx) <= DOCK_X_TOL and abs(dy) <= DOCK_Y_TOL) else 0.0
    aligned = 1.0 if align_err < 0.5 else 0.0
    stopped = 1.0 if spd < SPEED_TOL else 0.0
    done = inside * aligned * stopped
    completion_bonus = 4000.0 * done

    # ---------- 组件 4: 软接触惩罚（仅在接触且货箱速度突变时轻罚） ----------
    contact = next_obs[14]
    # 用货箱速度幅值作为"硬推"的间接证据，仅在接触时生效
    hard_push = contact * max(0.0, spd - 0.5) * max(0.0, spd - 0.5)
    soft_contact_penalty = -1.0 * hard_push

    # ---------- 组件 5: 越界惩罚（hinge，仅在接近边界时生效） ----------
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    oob = 0.0
    if cart_x > 0.92:
        oob += (cart_x - 0.92)
    if cart_x < -0.92:
        oob += (-0.92 - cart_x)
    if cart_y > 0.92:
        oob += (cart_y - 0.92)
    if cart_y < -0.92:
        oob += (-0.92 - cart_y)
    out_of_bounds_penalty = -4.0 * oob

    # ---------- 组件 6: 动作平滑（轻量） ----------
    action_smoothness = -0.05 * (action[0] * action[0] + action[1] * action[1])

    total_reward = (
        crate_progress
        + dock_quality
        + completion_bonus
        + soft_contact_penalty
        + out_of_bounds_penalty
        + action_smoothness
    )

    components = {
        "crate_progress": crate_progress,
        "dock_quality": dock_quality,
        "completion_bonus": completion_bonus,
        "soft_contact_penalty": soft_contact_penalty,
        "out_of_bounds_penalty": out_of_bounds_penalty,
        "action_smoothness": action_smoothness,
    }

    return (float(total_reward), components)
```