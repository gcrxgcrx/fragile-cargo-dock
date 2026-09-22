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

    # ---- Component A: Landing proxy (replaces generic goal_proximity) ----
    # Touch factor: 0 when airborne, up to 2 when both legs touch
    touch_bonus = left_contact + right_contact
    # Distance factor: close to 1 when near pad center
    dist_sq = x_pos**2 + y_pos**2
    dist_factor = 1.0 / (1.0 + dist_sq)
    # Speed factor: close to 1 when near-zero velocity (stable)
    speed_sq = x_vel**2 + y_vel**2 + 0.1 * ang_vel**2
    speed_factor = 1.0 / (1.0 + speed_sq)
    # Joint reward: product of three bounded (0..1 or 0..2) factors
    landing_proxy = 2.0 * touch_bonus * dist_factor * speed_factor

    # ---- Component B: Safe landing constraint (soft gate, reduced) ----
    height_gate = max(0.0, 1.0 - y_pos / 2.0) if y_pos < 2.0 else 0.0
    contact_gate = 0.2 * (left_contact + right_contact)
    landing_gate = height_gate + contact_gate

    # Motion cost still penalises heavy movement near ground, but coefficient is now -0.1
    motion_cost = x_vel**2 + y_vel**2 + body_angle**2 + 0.1 * ang_vel**2
    safe_landing_penalty = -0.1 * landing_gate * motion_cost

    # ---- Component C: Fuel efficiency ----
    fuel_penalty = -0.01 if action != 0 else 0.0

    # ---- Total reward ----
    total_reward = landing_proxy + safe_landing_penalty + fuel_penalty

    components = {
        "landing_proxy": landing_proxy,
        "safe_landing_penalty": safe_landing_penalty,
        "fuel_penalty": fuel_penalty
    }

    return float(total_reward), components