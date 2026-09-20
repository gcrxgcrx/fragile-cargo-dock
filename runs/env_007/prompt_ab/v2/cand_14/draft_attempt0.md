```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 观测解包（仅使用已声明维度）----
    crate_dx = next_obs[12]   # 货箱到泊位有符号 x 偏移 / 半宽
    crate_dy = next_obs[13]   # 货箱到泊位有符号 y 偏移 / 半高
    prev_dx = obs[12]
    prev_dy = obs[13]

    crate_vx = next_obs[8]    # / 3.0 m/s
    crate_vy = next_obs[9]    # / 3.0 m/s
    crate_cos = next_obs[10]
    crate_sin = next_obs[11]

    contact = next_obs[14]
    front_sensor = next_obs[15]
    left_sensor = next_obs[16]
    right_sensor = next_obs[17]

    drive = action[0]
    steer = action[1]

    # ---- 距离度量（用归一化偏移，避免开方误差放大）----
    prev_dist = (prev_dx * prev_dx + prev_dy * prev_dy) ** 0.5
    curr_dist = (crate_dx * crate_dx + crate_dy * crate_dy) ** 0.5

    # ---- 完成判据（严格取自环境事实：|dx|<=0.024, |dy|<=0.030, 朝向<30°, 速度<0.05）----
    pos_ok = 1.0 if (abs(crate_dx) <= 0.024 and abs(crate_dy) <= 0.030) else 0.0

    # 朝向误差：货箱朝向与泊位朝向（假定泊位朝 +x，即 heading=0）的夹角
    # 用 cos 误差近似：cos(theta_err) = crate_cos
    orient_ok = 1.0 if crate_cos >= 0.8660254 else 0.0   # cos(30°)

    # 货箱速度：obs[8], obs[9] 已除以 3.0 m/s，0.05 m/s -> 0.05/3.0 ≈ 0.01667
    crate_speed_norm = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    speed_ok = 1.0 if crate_speed_norm <= 0.01667 else 0.0

    docked_flag = pos_ok * orient_ok * speed_ok

    # ---- 组件 1：货箱向泊位推进（增量形式，避免悬停收割）----
    progress = prev_dist - curr_dist
    # 只奖励靠近，不惩罚远离（远离由其他信号约束），避免"惩罚必要动作"
    crate_to_dock_progress = 5.0 * max(0.0, progress)

    # ---- 组件 2：接近泊位后的精细对齐 shaping（门控，仅近距离激活）----
    # 门控因子：距离越近权重越大（用 1/(1+10*dist) 形式，距离 0 时最大）
    near_gate = 1.0 / (1.0 + 10.0 * curr_dist)

    # 朝向对齐 shaping（连续，用作 near_gate 内的辅助）
    orient_err = 1.0 - crate_cos   # cos=1 时误差 0，cos=0.866 时误差约 0.134
    orient_shaping = near_gate * max(0.0, 1.0 - orient_err * 3.0)

    # 泊位内低速 shaping（仅在已进入泊位容差区时激活）
    in_dock_gate = pos_ok
    speed_shaping = in_dock_gate * max(0.0, 1.0 - crate_speed_norm / 0.01667)

    crate_docking_quality = 0.3 * orient_shaping + 0.5 * speed_shaping

    # ---- 组件 3：完成事件大额奖励（一次性主导信号）----
    # 完成条件满足时给大额奖励，且要求连续保持（由环境自身判定，此处仅给即时信号）
    # 量级：B 需满足 10*B > 3*(过程组件上限*400)
    # 过程组件每步上限约 5.0*progress_max + 0.8 ≈ 5.8，*400*3 = 6960，故 B > 696
    # 取 B = 800 确保主导
    completion_bonus = 800.0 * docked_flag

    # ---- 组件 4：软接触惩罚（仅在接触时且相对速度大时轻罚）----
    # 用货箱速度作为间接冲量代理，接触时惩罚过大的货箱速度
    soft_contact_penalty = 0.0
    if contact > 0.5:
        excess = max(0.0, crate_speed_norm - 0.01667)
        soft_contact_penalty = -0.5 * excess

    # ---- 组件 5：越界 hinge 惩罚（小车位置接近边界）----
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    # 仓库半宽/半高归一化到 ±1，边界留 0.1 余量
    oob_penalty = 0.0
    oob_penalty += -2.0 * max(0.0, abs(cart_x) - 0.9)
    oob_penalty += -2.0 * max(0.0, abs(cart_y) - 0.9)

    # ---- 组件 6：动作平滑（轻量，避免高频抖动）----
    action_smoothness = -0.02 * (drive * drive + steer * steer)

    # ---- 组件 7：障碍接近惩罚（前/左/右传感器，hinge，仅接近时激活）----
    obstacle_penalty = 0.0
    obstacle_penalty += -0.3 * max(0.0, front_sensor - 0.8)
    obstacle_penalty += -0.3 * max(0.0, left_sensor - 0.8)
    obstacle_penalty += -0.3 * max(0.0, right_sensor - 0.8)

    # ---- 汇总 ----
    components = {
        "crate_to_dock_progress": crate_to_dock_progress,
        "crate_docking_quality": crate_docking_quality,
        "completion_bonus": completion_bonus,
        "soft_contact_penalty": soft_contact_penalty,
        "out_of_bounds_penalty": oob_penalty,
        "action_smoothness": action_smoothness,
        "obstacle_penalty": obstacle_penalty,
    }

    total_reward = (
        crate_to_dock_progress
        + crate_docking_quality
        + completion_bonus
        + soft_contact_penalty
        + oob_penalty
        + action_smoothness
        + obstacle_penalty
    )

    return (float(total_reward), components)
```