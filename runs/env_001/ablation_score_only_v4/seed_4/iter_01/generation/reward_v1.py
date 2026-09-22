def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next_obs signals
    x_pos = next_obs[0]   # horizontal position relative to target pad
    y_pos = next_obs[1]   # vertical position relative to pad (0 at pad)
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]   # 1.0 if contact, 0.0 otherwise
    right_contact = next_obs[7]

    # ---- Component A: Goal proximity ----
    # Encourage moving toward center (x=0, y=0)
    dist_sq = x_pos**2 + y_pos**2
    goal_proximity = 2.0 / (1.0 + dist_sq)

    # ---- Component B: Safe landing constraint ----
    # Gate activates when near ground or any leg touches
    # y_pos threshold: below 2.0 the gate gradually rises to 1.0 at y=0
    height_gate = max(0.0, 1.0 - y_pos / 2.0) if y_pos < 2.0 else 0.0
    contact_gate = 0.2 * (left_contact + right_contact)
    landing_gate = height_gate + contact_gate

    # Penalise excessive speed, tilt, and angular velocity when close to ground or in contact
    # Angular velocity is scaled down to be comparable
    motion_cost = x_vel**2 + y_vel**2 + body_angle**2 + 0.1 * ang_vel**2
    safe_landing_penalty = -0.5 * landing_gate * motion_cost

    # ---- Component C: Fuel efficiency ----
    # Small fixed cost for any engine use (actions 1,2,3)
    fuel_penalty = -0.01 if action != 0 else 0.0

    # ---- Total reward ----
    total_reward = goal_proximity + safe_landing_penalty + fuel_penalty

    components = {
        "goal_proximity": goal_proximity,
        "safe_landing_penalty": safe_landing_penalty,
        "fuel_penalty": fuel_penalty
    }

    return float(total_reward), components