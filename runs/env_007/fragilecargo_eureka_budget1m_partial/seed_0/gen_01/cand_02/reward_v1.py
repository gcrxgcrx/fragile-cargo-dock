def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------------- primary role: crate net progress toward the dock ----------------
    dist_prev = ((obs[12] * 5.0) ** 2 + (obs[13] * 4.0) ** 2) ** 0.5
    dist_next = ((next_obs[12] * 5.0) ** 2 + (next_obs[13] * 4.0) ** 2) ** 0.5
    d_prog = dist_prev - dist_next
    if d_prog > 0.15:
        d_prog = 0.15
    elif d_prog < -0.15:
        d_prog = -0.15
    crate_progress = 15.0 * d_prog

    # ---------------- secondary role: reach the crate before first contact ----------------
    rel_prev = ((obs[6] * 3.0) ** 2 + (obs[7] * 3.0) ** 2) ** 0.5
    rel_next = ((next_obs[6] * 3.0) ** 2 + (next_obs[7] * 3.0) ** 2) ** 0.5
    d_approach = rel_prev - rel_next
    if d_approach > 0.10:
        d_approach = 0.10
    elif d_approach < -0.10:
        d_approach = -0.10
    cart_crate_approach = 2.0 * d_approach

    # ---------------- milestone role: crate docked, slow, axis aligned ----------------
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    axis_double = abs(2.0 * next_obs[10] * next_obs[11])
    alignment_factor = max(0.0, 1.0 - axis_double / 0.8660254)
    distance_factor = max(0.0, 1.0 - dist_next / 0.35)
    speed_factor = max(0.0, 1.0 - crate_speed / 0.12)
    docked_settle = 9.0 * distance_factor * speed_factor * alignment_factor

    # ---------------- approach gate: the crate should arrive slow ----------------
    approach_gate = max(0.0, 1.0 - dist_next / 1.5)
    settle_speed_penalty = -0.5 * approach_gate * crate_speed

    # ---------------- fragile handling guard: no hard hits on the crate ----------------
    contact_penalty = 0.0
    if next_obs[14] > 0.5:
        cart_vx = next_obs[4] * 3.0 * next_obs[2]
        cart_vy = next_obs[4] * 3.0 * next_obs[3]
        crate_vx = next_obs[8] * 3.0
        crate_vy = next_obs[9] * 3.0
        rel_vx = cart_vx - crate_vx
        rel_vy = cart_vy - crate_vy
        rel_speed = (rel_vx ** 2 + rel_vy ** 2) ** 0.5
        over_speed = rel_speed - 0.30
        if over_speed > 0.0:
            contact_penalty = -1.5 * over_speed ** 2

    # ---------------- keep cart and crate on the warehouse floor ----------------
    boundary_penalty = 0.0
    cart_x_ratio = abs(next_obs[0])
    cart_y_ratio = abs(next_obs[1])
    if cart_x_ratio > 0.90:
        boundary_penalty -= 2.0 * (cart_x_ratio - 0.90) ** 2
    if cart_y_ratio > 0.90:
        boundary_penalty -= 2.0 * (cart_y_ratio - 0.90) ** 2

    crate_x_world = (next_obs[0] * 5.0
                     + next_obs[2] * next_obs[6] * 3.0
                     - next_obs[3] * next_obs[7] * 3.0)
    crate_y_world = (next_obs[1] * 4.0
                     + next_obs[3] * next_obs[6] * 3.0
                     + next_obs[2] * next_obs[7] * 3.0)
    if abs(crate_x_world) > 4.5:
        boundary_penalty -= 2.0 * (abs(crate_x_world) - 4.5) ** 2
    if abs(crate_y_world) > 3.5:
        boundary_penalty -= 2.0 * (abs(crate_y_world) - 3.5) ** 2

    components = {
        "crate_progress_toward_dock": crate_progress,
        "cart_crate_approach": cart_crate_approach,
        "docked_settle": docked_settle,
        "settle_speed_penalty": settle_speed_penalty,
        "fragile_handling_penalty": contact_penalty,
        "boundary_penalty": boundary_penalty,
    }
    total_reward = (crate_progress
                    + cart_crate_approach
                    + docked_settle
                    + settle_speed_penalty
                    + contact_penalty
                    + boundary_penalty)
    return float(total_reward), components