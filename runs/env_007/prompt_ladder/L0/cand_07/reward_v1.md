```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------------- 位置与距离度量 ----------------
    # 货箱到泊位的归一化有符号偏移
    dx = obs[12]
    dy = obs[13]
    dist = (dx * dx + dy * dy) ** 0.5

    dx_n = next_obs[12]
    dy_n = next_obs[13]
    dist_next = (dx_n * dx_n + dy_n * dy_n) ** 0.5

    # ---------------- 1. 货箱向泊位推进（主进度信号） ----------------
    # improvement delta：距离缩短为正，避免悬停陷阱
    progress = (dist - dist_next) * 5.0

    # ---------------- 2. 货箱停靠质量（位置 / 朝向 / 速度） ----------------
    # 位置接近度：距离越小越接近 1
    pos_prox = 1.0 / (1.0 + 20.0 * dist)

    # 朝向误差（货箱朝向与泊位对齐，泊位朝向视为 0 轴）
    crate_angle = (next_obs[11] * next_obs[11] + next_obs[10] * next_obs[10]) ** 0.5
    if crate_angle > 1e-6:
        ang_err = (next_obs[11] / crate_angle)
        # |sin(angle_error)| ~ 归一化朝向偏差
        if ang_err < 0.0:
            ang_err = -ang_err
    else:
        ang_err = 1.0
    # 朝向对齐因子：误差 30 度以内（sin30=0.5）给高分
    align = 1.0 / (1.0 + 6.0 * ang_err)

    # 货箱速度（世界系）
    cvx = next_obs[8]
    cvy = next_obs[9]
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    # 速度静止因子：0.05 m/s 对应归一化约 0.0167
    still = 1.0 / (1.0 + 60.0 * crate_speed)

    # 联合停靠质量（几何平均，避免塌缩）
    dock_quality = (pos_prox * align * still) ** (1.0 / 3.0)
    # 仅在接近泊位时给足权重，用 pos_prox 作为门
    dock_reward = 2.0 * dock_quality * pos_prox

    # ---------------- 3. 接近泊位时的速度抑制 ----------------
    # 货箱接近泊位（dist < 0.15）时抑制高速
    near_gate = 1.0 / (1.0 + 40.0 * dist)
    speed_excess = crate_speed - 0.05
    if speed_excess < 0.0:
        speed_excess = 0.0
    speed_penalty = -1.5 * near_gate * (speed_excess ** 2)

    # ---------------- 4. 轻柔接触（间接推断硬碰撞） ----------------
    # 接触时若货箱速度突增，视为硬冲击风险
    contact = obs[14]
    prev_speed = (obs[8] * obs[8] + obs[9] * obs[9]) ** 0.5
    speed_jump = crate_speed - prev_speed
    if speed_jump < 0.0:
        speed_jump = 0.0
    hard_proxy = contact * speed_jump
    soft_contact_penalty = -3.0 * (hard_proxy ** 2)

    # ---------------- 5. 越界防护（hinge 形式） ----------------
    # 小车位置归一化到 [-1, 1]，接近 ±1 时惩罚
    cart_x = obs[0]
    cart_y = obs[1]
    ax = cart_x
    if ax < 0.0:
        ax = -ax
    ay = cart_y
    if ay < 0.0:
        ay = -ay
    bound_penalty = 0.0
    if ax > 0.85:
        bound_penalty -= 2.0 * (ax - 0.85)
    if ay > 0.85:
        bound_penalty -= 2.0 * (ay - 0.85)

    # 货箱越界风险：泊位偏移过大意味着货箱可能靠近边界
    crate_bound = 0.0
    if dist > 0.9:
        crate_bound = -1.0 * (dist - 0.9)

    # ---------------- 6. 动作平滑（轻量） ----------------
    act_sq = action[0] * action[0] + action[1] * action[1]
    smooth_penalty = -0.02 * act_sq

    # ---------------- 汇总 ----------------
    components = {}
    components["crate_progress"] = progress
    components["dock_quality"] = dock_reward
    components["speed_near_dock"] = speed_penalty
    components["soft_contact"] = soft_contact_penalty
    components["boundary"] = bound_penalty + crate_bound
    components["action_smooth"] = smooth_penalty

    total = (
        progress
        + dock_reward
        + speed_penalty
        + soft_contact_penalty
        + bound_penalty
        + crate_bound
        + smooth_penalty
    )

    if total > 20.0:
        total = 20.0
    if total < -20.0:
        total = -20.0

    return float(total), components
```