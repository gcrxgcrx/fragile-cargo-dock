def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 1) 主进度：货箱向 dock 的净位移（delta 形式，抑制悬停） ----------
    dist_prev = ((obs[12] * 5.0) ** 2 + (obs[13] * 4.0) ** 2) ** 0.5
    dist_next = ((next_obs[12] * 5.0) ** 2 + (next_obs[13] * 4.0) ** 2) ** 0.5
    progress = dist_prev - dist_next
    if progress > 0.2:
        progress = 0.2
    elif progress < -0.2:
        progress = -0.2
    crate_progress = 15.0 * progress

    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    align_raw = abs(2.0 * next_obs[10] * next_obs[11])  # = |sin(2*theta)|

    # ---------- 2) 收紧后的接近-沉降塑形（阈值更紧、权重降低，防悬停刷分） ----------
    close_factor = max(0.0, 1.0 - dist_next / 0.35)
    slow_factor = max(0.0, 1.0 - crate_speed / 0.12)
    align_factor = max(0.0, 1.0 - align_raw / 0.8660254)
    joint = (close_factor * slow_factor * align_factor) ** (1.0 / 3.0)
    docking_settle = 1.5 * joint

    # ---------- 3) 真正的停靠里程碑：紧邻 + 近静止 + 对齐的联合条件（连续化） ----------
    close_t = max(0.0, 1.0 - dist_next / 0.18)
    slow_t = max(0.0, 1.0 - crate_speed / 0.06)
    align_t = max(0.0, 1.0 - align_raw / 0.5)
    tight = (close_t * slow_t * align_t) ** (1.0 / 3.0)
    docking_completion = 6.0 * (tight ** 2)

    # ---------- 4) 轻拿轻放：仅对高速接触做 hinge 惩罚 ----------
    contact_penalty = 0.0
    if next_obs[14] > 0.5:
        cart_vx = next_obs[4] * 3.0 * next_obs[2]
        cart_vy = next_obs[4] * 3.0 * next_obs[3]
        crate_vx = next_obs[8] * 3.0
        crate_vy = next_obs[9] * 3.0
        rel_speed = ((cart_vx - crate_vx) ** 2 + (cart_vy - crate_vy) ** 2) ** 0.5
        if rel_speed > 0.3:
            contact_penalty = -1.5 * (rel_speed - 0.3) ** 2

    # ---------- 5) 边界防护：小车与货箱都不许离开地板 ----------
    boundary_penalty = 0.0
    cart_x_ratio = abs(next_obs[0])
    cart_y_ratio = abs(next_obs[1])
    if cart_x_ratio > 0.9:
        boundary_penalty -= 2.0 * (cart_x_ratio - 0.9) ** 2
    if cart_y_ratio > 0.9:
        boundary_penalty -= 2.0 * (cart_y_ratio - 0.9) ** 2

    cart_x_world = next_obs[0] * 5.0
    cart_y_world = next_obs[1] * 4.0
    heading_cos = next_obs[2]
    heading_sin = next_obs[3]
    rel_x = next_obs[6] * 3.0
    rel_y = next_obs[7] * 3.0
    crate_x_world = cart_x_world + heading_cos * rel_x - heading_sin * rel_y
    crate_y_world = cart_y_world + heading_sin * rel_x + heading_cos * rel_y
    if abs(crate_x_world) > 4.5:
        boundary_penalty -= 2.0 * (abs(crate_x_world) - 4.5) ** 2
    if abs(crate_y_world) > 3.5:
        boundary_penalty -= 2.0 * (abs(crate_y_world) - 3.5) ** 2

    components = {
        "crate_progress_toward_dock": crate_progress,
        "docking_settle": docking_settle,
        "docking_completion": docking_completion,
        "fragile_handling_penalty": contact_penalty,
        "boundary_penalty": boundary_penalty,
    }
    total_reward = crate_progress + docking_settle + docking_completion + contact_penalty + boundary_penalty
    return float(total_reward), components