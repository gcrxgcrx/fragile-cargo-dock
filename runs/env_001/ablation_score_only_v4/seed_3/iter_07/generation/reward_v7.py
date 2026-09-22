def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v2: landing_bonus → landing_quality (dense per-step); gate orientation_penalty.
    """
    # --- Unpack observations ---
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx_pos, ny_pos = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_body_angle = next_obs[4]
    n_angular_vel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # --- Component A: delta_distance ---
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (nx_pos**2 + ny_pos**2) ** 0.5
    delta_distance = current_distance - next_distance
    progress_reward = 10.0 * delta_distance

    # --- Component B: bounded_distance_shaping ---
    distance_shaping = 2.0 * (1.0 / (1.0 + 0.5 * next_distance))

    # --- Component C: velocity_hinge_constraint ---
    h_speed = abs(nx_vel)
    h_penalty = max(0.0, h_speed - 2.0)
    v_speed = ny_vel
    v_down_penalty = max(0.0, -v_speed - 1.5)
    v_up_penalty = max(0.0, v_speed - 1.0)
    angular_penalty = max(0.0, abs(n_angular_vel) - 0.8)
    velocity_constraint = -0.5 * (h_penalty + v_down_penalty + v_up_penalty + angular_penalty)

    # --- Component D: orientation_stabilization (gated) ---
    # Gate: average contact of next legs, 0 if no contact, 1 if both contact
    contact_gate = (n_left_contact + n_right_contact) / 2.0
    # When after contact, disable orientation penalty to avoid conflict with landing_quality
    orientation_penalty = (1.0 - contact_gate) * (-0.3 * (n_body_angle**2) - 0.2 * (n_angular_vel**2))

    # --- Component E: fuel_efficiency ---
    if action == 0:
        fuel_penalty = 0.0
    elif action == 2:
        fuel_penalty = -0.08
    else:
        fuel_penalty = -0.05

    # --- Component F: landing_quality (dense per-step) ---
    # Continuous bounded factors for safe landing status
    speed_norm = (nx_vel**2 + ny_vel**2) ** 0.5
    # Scale such that perfect zero gives 1.0, and at landing thresholds gives ~0.0
    safe_speed_factor = max(0.0, 1.0 - speed_norm / 0.4)   # 0 at 0.4 speed, 1 at 0
    safe_angle_factor = max(0.0, 1.0 - abs(n_body_angle) / 0.2)  # 0 at 0.2 rad
    safe_spin_factor = max(0.0, 1.0 - abs(n_angular_vel) / 0.2)  # 0 at 0.2 rad/s

    # Landing quality = product of factor * contact_gate (so only active when legs touch)
    landing_quality = 0.5 * contact_gate * safe_speed_factor * safe_angle_factor * safe_spin_factor

    # --- Assemble total reward ---
    total_reward = (
        progress_reward +
        distance_shaping +
        velocity_constraint +
        orientation_penalty +
        fuel_penalty +
        landing_quality
    )

    components = {
        'progress_reward': progress_reward,
        'distance_shaping': distance_shaping,
        'velocity_constraint': velocity_constraint,
        'orientation_penalty': orientation_penalty,
        'fuel_penalty': fuel_penalty,
        'landing_quality': landing_quality
    }

    return float(total_reward), components