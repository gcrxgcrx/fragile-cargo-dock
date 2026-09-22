def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations (origin at target platform center/height)
    x_pos, y_pos, x_vel, y_vel, body_angle, ang_vel, l_contact, r_contact = obs
    nx_pos, ny_pos, nx_vel, ny_vel, n_angle, n_ang_vel, nl_contact, nr_contact = next_obs

    # Distance to target
    current_dist = (x_pos**2 + y_pos**2) ** 0.5
    next_dist    = (nx_pos**2 + ny_pos**2) ** 0.5

    # Speed magnitude
    current_speed = (x_vel**2 + y_vel**2) ** 0.5
    next_speed    = (nx_vel**2 + ny_vel**2) ** 0.5

    # ---------- Component A: potential‑based shaping for approach + deceleration ----------
    w_dist = 1.0
    w_speed = 0.5
    phi_current = -(w_dist * current_dist + w_speed * current_speed)
    phi_next    = -(w_dist * next_dist    + w_speed * next_speed)
    progress = phi_next - phi_current   # positive when distance shrinks or speed reduces

    # ---------- Component B: upright orientation penalty ----------
    orientation_penalty = -0.1 * (n_angle ** 2) - 0.05 * (n_ang_vel ** 2)

    # ---------- Component C: fuel cost ----------
    if action == 0:                     # no engine
        fuel_cost = 0.0
    elif action == 2:                   # main engine (most costly)
        fuel_cost = -0.1
    else:                               # orientation engines (1 or 3)
        fuel_cost = -0.05

    # ---------- Total ----------
    total = progress + orientation_penalty + fuel_cost

    components = {
        "progress": progress,
        "orientation_penalty": orientation_penalty,
        "fuel_cost": fuel_cost
    }
    return float(total), components