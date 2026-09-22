def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v4: safe_approach (landing readiness throughout final descent) replaces
                approach_and_land (which only activated safety on contact).
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

    # --- Component A: delta_distance (unchanged) ---
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (nx_pos**2 + ny_pos**2) ** 0.5
    delta_distance = current_distance - next_distance
    progress_reward = 10.0 * delta_distance

    # --- Component B: bounded_distance_shaping (unchanged) ---
    distance_shaping = 2.0 * (1.0 / (1.0 + 0.5 * next_distance))

    # --- Component C: velocity_hinge_constraint (unchanged) ---
    h_speed = abs(nx_vel)
    h_penalty = max(0.0, h_speed - 2.0)
    v_speed = ny_vel
    v_down_penalty = max(0.0, -v_speed - 1.5)
    v_up_penalty = max(0.0, v_speed - 1.0)
    angular_penalty = max(0.0, abs(n_angular_vel) - 0.8)
    velocity_constraint = -0.5 * (h_penalty + v_down_penalty + v_up_penalty + angular_penalty)

    # --- Component D: orientation_stabilization (gated, unchanged) ---
    contact_gate = (n_left_contact + n_right_contact) / 2.0
    orientation_penalty = (1.0 - contact_gate) * (-0.3 * (n_body_angle**2) - 0.2 * (n_angular_vel**2))

    # --- Component E: efficiency_gate (unchanged) ---
    proximity_gate = max(0.0, 1.0 - next_distance / 1.5)
    if action == 0:
        efficiency_penalty = 0.0
    elif action == 2:
        efficiency_penalty = -0.08 * proximity_gate
    else:
        efficiency_penalty = -0.03 * proximity_gate

    # --- Component F: safe_approach (replaces approach_and_land) ---
    # Continuous landing-readiness signal that activates as distance decreases.
    # Uses geometric mean of three safety factors, blended with proximity.
    
    # Proximity weight: ramps from 0 at dist >= 5 to 1 at dist = 0
    proximity_weight = max(0.0, 1.0 - next_distance / 5.0)
    
    # Safety factors (bounded [0, 1], soft ramp from "dangerous" to "safe")
    speed_norm = (nx_vel**2 + ny_vel**2) ** 0.5
    # Safe speed: target < 0.5, dangerous > 1.5
    safe_speed = max(0.0, 1.0 - speed_norm / 1.5)
    # Safe angle: target < 0.15, dangerous > 0.5
    safe_angle = max(0.0, 1.0 - abs(n_body_angle) / 0.5)
    # Safe spin: target < 0.15, dangerous > 0.5
    safe_spin = max(0.0, 1.0 - abs(n_angular_vel) / 0.5)
    
    # Geometric mean: prevents total collapse when one factor is slightly off
    safe_product = safe_speed * safe_angle * safe_spin
    safe_geo_mean = safe_product ** (1.0 / 3.0) if safe_product > 0.0 else 0.0
    
    # Safe approach: proximity_weight gates the landing-readiness signal
    # When far away, agent focuses on approaching; as it gets close, safety matters.
    safe_approach = proximity_weight * safe_geo_mean * 2.0

    # --- Assemble total reward ---
    total_reward = (
        progress_reward +
        distance_shaping +
        velocity_constraint +
        orientation_penalty +
        efficiency_penalty +
        safe_approach
    )

    components = {
        'progress_reward': progress_reward,
        'distance_shaping': distance_shaping,
        'velocity_constraint': velocity_constraint,
        'orientation_penalty': orientation_penalty,
        'efficiency_penalty': efficiency_penalty,
        'safe_approach': safe_approach
    }

    return float(total_reward), components