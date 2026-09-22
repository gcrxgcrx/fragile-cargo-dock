def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 0. 回合级持久状态（仅用函数属性保存，用于一次性事件与连续计数） ----------
    if not hasattr(compute_reward, "_init_done"):
        compute_reward._init_done = True
        compute_reward._dock_enter_issued = False
        compute_reward._stable_steps = 0
        compute_reward._success_issued = False
        compute_reward._hard_hits = 0
        compute_reward._failure_issued = False
        compute_reward._prev_contact = 0.0

    # time_fraction 回退 => 本步是新一轮 episode 的第一步，重置回合状态
    if next_obs[18] < obs[18] - 1e-9:
        compute_reward._dock_enter_issued = False
        compute_reward._stable_steps = 0
        compute_reward._success_issued = False
        compute_reward._hard_hits = 0
        compute_reward._failure_issued = False
        compute_reward._prev_contact = 0.0

    # ---------- 1. approach_cargo：小车->货箱距离本帧缩短量（米，势函数差分，有符号） ----------
    cart_crate_now = 3.0 * ((obs[6] * obs[6] + obs[7] * obs[7]) ** 0.5)
    cart_crate_next = 3.0 * ((next_obs[6] * next_obs[6] + next_obs[7] * next_obs[7]) ** 0.5)
    approach_cargo = 1.0 * (cart_crate_now - cart_crate_next)

    # ---------- 2. progress：货箱->泊位距离本帧缩短量（米，势函数差分，有符号） ----------
    dock_now = (((obs[12] * 5.0) ** 2) + ((obs[13] * 4.0) ** 2)) ** 0.5
    dock_next = (((next_obs[12] * 5.0) ** 2) + ((next_obs[13] * 4.0) ** 2)) ** 0.5
    progress = 1.0 * (dock_now - dock_next)

    # ---------- 3. dock_enter：货箱首次完全进入泊位容差，一次性发放 ----------
    in_dock = (abs(next_obs[12]) <= 0.024) and (abs(next_obs[13]) <= 0.030)
    dock_enter = 0.0
    if in_dock and not compute_reward._dock_enter_issued:
        dock_enter = 5.0
        compute_reward._dock_enter_issued = True

    # ---------- 4. roughness / hard_hit：接触闭合速度代理（冲量不可读，用可观测代理） ----------
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    closing = obs[4] * 3.0 - (obs[8] * 3.0 * obs[2] + obs[9] * 3.0 * obs[3])
    if closing < 0.0:
        closing = 0.0
    impact_speed = closing * contact
    roughness = -0.02 * impact_speed

    hard_hit = 0.0
    if contact > 0.5 and compute_reward._prev_contact <= 0.5 and closing > 1.0:
        hard_hit = -0.5
        compute_reward._hard_hits += 1
    compute_reward._prev_contact = contact

    # ---------- 5. action_cost / time_cost ----------
    action_cost = -0.0005 * (action[0] * action[0] + action[1] * action[1])
    time_cost = -0.002

    # ---------- 6. terminal_success：入位 + 朝向对齐 + 低速，连续 10 步 ----------
    cos_h = next_obs[10]
    sin_h = next_obs[11]
    aligned = abs(cos_h * sin_h) < 0.433          # 方形货箱 90° 对称：|c*s|=0.5*|sin(2*err)|，30° 对应 0.433
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    settled = in_dock and aligned and (crate_speed < 0.05)
    if settled:
        compute_reward._stable_steps += 1
    else:
        compute_reward._stable_steps = 0

    terminal_success = 0.0
    if compute_reward._stable_steps >= 10 and not compute_reward._success_issued:
        terminal_success = 300.0
        compute_reward._success_issued = True

    # ---------- 7. terminal_failure：小车/货箱越界，或累计硬冲击 >= 3 ----------
    crate_x = next_obs[0] * 5.0 + cos_h * (next_obs[6] * 3.0) - sin_h * (next_obs[7] * 3.0)
    crate_y = next_obs[1] * 4.0 + sin_h * (next_obs[6] * 3.0) + cos_h * (next_obs[7] * 3.0)
    out_of_bounds = (abs(next_obs[0]) > 1.0) or (abs(next_obs[1]) > 1.0) or (abs(crate_x) > 5.0) or (abs(crate_y) > 4.0)

    terminal_failure = 0.0
    if (out_of_bounds or compute_reward._hard_hits >= 3) and not compute_reward._failure_issued:
        terminal_failure = -100.0
        compute_reward._failure_issued = True

    total_reward = (
        approach_cargo
        + progress
        + dock_enter
        + roughness
        + action_cost
        + time_cost
        + hard_hit
        + terminal_success
        + terminal_failure
    )

    components = {
        "approach_cargo": float(approach_cargo),
        "progress": float(progress),
        "dock_enter": float(dock_enter),
        "roughness": float(roughness),
        "action_cost": float(action_cost),
        "time_cost": float(time_cost),
        "hard_hit": float(hard_hit),
        "terminal_success": float(terminal_success),
        "terminal_failure": float(terminal_failure),
    }

    return float(total_reward), components