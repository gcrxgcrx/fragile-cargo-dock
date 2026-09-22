def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observation after action
    x_err = next_obs[0]   # horizontal position error relative to pad center
    y_err = next_obs[1]   # vertical position error (height above pad)
    vx = next_obs[2]      # horizontal velocity
    vy = next_obs[3]      # vertical velocity
    angle = next_obs[4]   # body tilt
    ang_vel = next_obs[5] # angular velocity
    left_contact = next_obs[6]  # left leg touch (0 or 1)
    right_contact = next_obs[7] # right leg touch (0 or 1)

    # ---------- Role 1: proximity_to_target (main learning signal) ----------
    dist = (x_err**2 + y_err**2)**0.5
    # Bounded negative distance to avoid extreme negative values at long range
    proximity_reward = -2.0 * (dist / (1.0 + dist))

    # ---------- Role 2: soft_landing_dynamics (stability constraint) ----------
    # Gate: activate when vertical error is small (close to landing zone)
    landing_threshold = 1.0
    gate = max(0.0, 1.0 - abs(y_err) / landing_threshold)
    # Quadratic penalty on velocities, tilt and angular velocity
    dynamics_cost = vx**2 + vy**2 + angle**2 + ang_vel**2
    soft_landing_penalty = -0.5 * gate * dynamics_cost

    # ---------- Role 3: safe_contact_encouragement (soft task completion proxy) ----------
    both_legs_contact = left_contact * right_contact  # 1 if both legs touch
    # Tolerance windows for safe landing conditions
    x_tol = 0.3
    y_tol = 0.3
    v_tol = 0.2
    angle_tol = 0.1
    angvel_tol = 0.1
    # Continuous bounded factors (1 = perfect, 0 = outside tolerance)
    factor_x = max(0.0, 1.0 - abs(x_err) / x_tol)
    factor_y = max(0.0, 1.0 - abs(y_err) / y_tol)
    factor_vx = max(0.0, 1.0 - abs(vx) / v_tol)
    factor_vy = max(0.0, 1.0 - abs(vy) / v_tol)
    factor_angle = max(0.0, 1.0 - abs(angle) / angle_tol)
    factor_angvel = max(0.0, 1.0 - abs(ang_vel) / angvel_tol)
    # Mean of factors to avoid product collapse and maintain non‑zero gradient
    mean_safety = (factor_x + factor_y + factor_vx + factor_vy + factor_angle + factor_angvel) / 6.0
    safe_contact_bonus = both_legs_contact * mean_safety * 2.0

    total_reward = proximity_reward + soft_landing_penalty + safe_contact_bonus

    components = {
        "proximity_reward": proximity_reward,
        "soft_landing_penalty": soft_landing_penalty,
        "safe_contact_bonus": safe_contact_bonus
    }
    return float(total_reward), components