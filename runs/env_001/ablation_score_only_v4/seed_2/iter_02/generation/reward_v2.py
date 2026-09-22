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

    # ---------- Component A: potential-based shaping for approach + deceleration ----------
    w_dist = 1.0
    w_speed = 0.5
    phi_current = -(w_dist * current_dist + w_speed * current_speed)
    phi_next    = -(w_dist * next_dist    + w_speed * next_speed)
    progress = phi_next - phi_current

    # ---------- Component B: upright orientation penalty ----------
    orientation_penalty = -0.1 * (n_angle ** 2) - 0.05 * (n_ang_vel ** 2)

    # ---------- Component C: fuel cost ----------
    if action == 0:
        fuel_cost = 0.0
    elif action == 2:
        fuel_cost = -0.1
    else:   # 1 or 3
        fuel_cost = -0.05

    # ---------- Component D: crash prevention ----------
    # Penalize high downward speed when close to target platform.
    crash_risk = max(0.0, -ny_vel - 0.1)  # excess downward speed beyond 0.1
    proximity = max(0.0, 1.0 - next_dist / 0.3)  # linear ramp within 0.3 radius
    crash_prevention = -0.05 * crash_risk * proximity

    # ---------- Total ----------
    total = progress + orientation_penalty + fuel_cost + crash_prevention

    components = {
        "progress": progress,
        "orientation_penalty": orientation_penalty,
        "fuel_cost": fuel_cost,
        "crash_prevention": crash_prevention
    }
    return float(total), components