def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next_obs signals
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---- Component A: Dense landing progress (hinged, always positive gradient) ----
    # Position: close to pad center
    dist = (x_pos**2 + y_pos**2) ** 0.5
    position_factor = 1.0 / (1.0 + dist)  # [0,1], decreasing with distance
    
    # Velocity: prefer low speed
    speed = (x_vel**2 + y_vel**2 + 0.1 * ang_vel**2) ** 0.5
    speed_factor = 1.0 / (1.0 + speed)  # [0,1], decreasing with speed
    
    # Orientation: prefer upright
    angle_factor = max(0.0, 1.0 - abs(body_angle) / 3.14159265)  # [0,1], 1=perfectly upright
    
    # Contact: bonus for legs on ground
    contact_bonus = 0.3 * (left_contact + right_contact)  # [0, 0.6]
    
    # Combined landing proxy: weighted sum with contact as additive bonus
    landing_proxy = 1.5 * position_factor * speed_factor * angle_factor + contact_bonus

    # ---- Component B: Safe landing penalty (greatly reduced) ----
    # Gate: active only when near ground or in contact
    height_gate = max(0.0, 1.0 - y_pos / 2.0) if y_pos < 2.0 else 0.0
    contact_gate = 0.2 * (left_contact + right_contact)
    landing_gate = height_gate + contact_gate
    
    # Motion cost near ground
    motion_cost = x_vel**2 + y_vel**2 + body_angle**2 + 0.1 * ang_vel**2
    safe_landing_penalty = -0.01 * landing_gate * motion_cost

    # ---- Total reward ----
    total_reward = landing_proxy + safe_landing_penalty

    components = {
        "landing_proxy": landing_proxy,
        "safe_landing_penalty": safe_landing_penalty
    }

    return float(total_reward), components