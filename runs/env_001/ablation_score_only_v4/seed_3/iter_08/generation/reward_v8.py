def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v3: landing_quality → approach_and_land (dense proximity + safe landing);
                fuel_efficiency → efficiency_gate (only penalize non-essential actions).
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

    # --- Component A: delta_distance (keep) ---
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (nx_pos**2 + ny_pos**2) ** 0.5
    delta_distance = current_distance - next_distance
    progress_reward = 10.0 * delta_distance

    # --- Component B: bounded_distance_shaping (keep) ---
    distance_shaping = 2.0 * (1.0 / (1.0 + 0.5 * next_distance))

    # --- Component C: velocity_hinge_constraint (keep) ---
    h_speed = abs(nx_vel)
    h_penalty = max(0.0, h_speed - 2.0)
    v_speed = ny_vel
    v_down_penalty = max(0.0, -v_speed - 1.5)
    v_up_penalty = max(0.0, v_speed - 1.0)
    angular_penalty = max(0.0, abs(n_angular_vel) - 0.8)
    velocity_constraint = -0.5 * (h_penalty + v_down_penalty + v_up_penalty + angular_penalty)

    # --- Component D: orientation_stabilization (gated, keep) ---
    contact_gate = (n_left_contact + n_right_contact) / 2.0
    orientation_penalty = (1.0 - contact_gate) * (-0.3 * (n_body_angle**2) - 0.2 * (n_angular_vel**2))

    # --- Component E: efficiency_gate (replacing fuel_penalty) ---
    # Penalize main engine and orientation engines only when agent is close to platform
    # and doesn't need aggressive thrust — promote efficient final approach.
    proximity_gate = max(0.0, 1.0 - next_distance / 1.5)  # 0 at dist>=1.5, 1 at dist=0
    if action == 0:
        efficiency_penalty = 0.0
    elif action == 2:
        efficiency_penalty = -0.08 * proximity_gate
    else:
        efficiency_penalty = -0.03 * proximity_gate

    # --- Component F: approach_and_land (replacing landing_quality) ---
    # Proximity factor: exponential decay, gives gradient even when far away
    proximity_factor = 2.718281828 ** (-2.0 * next_distance)  # 1 at dist=0, ~0.135 at dist=1
    
    # Safe landing factors (geometric mean, only active on contact to avoid noise)
    speed_norm = (nx_vel**2 + ny_vel**2) ** 0.5
    safe_speed_factor = max(0.0, 1.0 - speed_norm / 0.4)
    safe_angle_factor = max(0.0, 1.0 - abs(n_body_angle) / 0.2)
    safe_spin_factor = max(0.0, 1.0 - abs(n_angular_vel) / 0.2)
    
    # Geometric mean of safety factors (no collapse if one factor is near 0)
    safe_factors_product = safe_speed_factor * safe_angle_factor * safe_spin_factor
    safe_geo_mean = safe_factors_product ** (1.0 / 3.0) if safe_factors_product > 0 else 0.0
    
    # Blend: proximity gives continuous signal; when in contact, safety improves it
    approach_and_land = proximity_factor * (1.0 + 2.0 * contact_gate * safe_geo_mean)

    # --- Assemble total reward ---
    total_reward = (
        progress_reward +
        distance_shaping +
        velocity_constraint +
        orientation_penalty +
        efficiency_penalty +
        0.8 * approach_and_land
    )

    components = {
        'progress_reward': progress_reward,
        'distance_shaping': distance_shaping,
        'velocity_constraint': velocity_constraint,
        'orientation_penalty': orientation_penalty,
        'efficiency_penalty': efficiency_penalty,
        'approach_and_land': 0.8 * approach_and_land
    }

    return float(total_reward), components