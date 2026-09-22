def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_pos, y_pos, x_vel, y_vel, body_angle, ang_vel, l_contact, r_contact = obs
    nx_pos, ny_pos, nx_vel, ny_vel, n_angle, n_ang_vel, nl_contact, nr_contact = next_obs

    current_dist = (x_pos**2 + y_pos**2) ** 0.5
    next_dist    = (nx_pos**2 + ny_pos**2) ** 0.5

    current_speed = (x_vel**2 + y_vel**2) ** 0.5
    next_speed    = (nx_vel**2 + ny_vel**2) ** 0.5

    w_dist = 1.0
    w_speed = 0.5
    phi_current = -(w_dist * current_dist + w_speed * current_speed)
    phi_next    = -(w_dist * next_dist    + w_speed * next_speed)
    progress = phi_next - phi_current

    orientation_penalty = -0.1 * (n_angle ** 2) - 0.05 * (n_ang_vel ** 2)

    if action == 0:
        fuel_cost = 0.0
    elif action == 2:
        fuel_cost = -0.1
    else:
        fuel_cost = -0.05

    crash_risk = max(0.0, -ny_vel - 0.1)
    proximity = max(0.0, 1.0 - next_dist / 0.3)
    crash_prevention = -0.05 * crash_risk * proximity

    # New landing progress reward
    contact_factor = min(nl_contact, nr_contact)  # requires both legs in contact
    dist_factor = max(0.0, 1.0 - next_dist / 0.3)
    speed_factor = max(0.0, 1.0 - next_speed / 0.5)
    landing_reward = 0.05 * contact_factor * dist_factor * speed_factor

    total = progress + orientation_penalty + fuel_cost + crash_prevention + landing_reward

    components = {
        "progress": progress,
        "orientation_penalty": orientation_penalty,
        "fuel_cost": fuel_cost,
        "crash_prevention": crash_prevention,
        "landing_reward": landing_reward
    }
    return float(total), components