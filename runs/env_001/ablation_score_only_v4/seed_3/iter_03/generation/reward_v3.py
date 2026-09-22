def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack previous and next observation
    x_err_prev = obs[0]
    y_err_prev = obs[1]
    dist_prev = (x_err_prev**2 + y_err_prev**2)**0.5

    x_err = next_obs[0]
    y_err = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    dist = (x_err**2 + y_err**2)**0.5

    # ---------- Role 1: proximity_to_target (improvement reward) ----------
    # Transform to improvement over step: reward approaching, penalise moving away
    f_prev = dist_prev / (1.0 + dist_prev)
    f_next = dist / (1.0 + dist)
    proximity_reward = -5.0 * (f_next - f_prev)

    # ---------- Role 2: soft_landing_dynamics (stability constraint) ----------
    landing_threshold = 2.0
    gate = max(0.0, 1.0 - abs(y_err) / landing_threshold)
    d_vx = vx**2 / (1.0 + vx**2)
    d_vy = vy**2 / (1.0 + vy**2)
    d_angle = angle**2 / (1.0 + angle**2)
    d_angvel = ang_vel**2 / (1.0 + ang_vel**2)
    dynamics_cost = d_vx + d_vy + d_angle + d_angvel
    soft_landing_penalty = -0.5 * gate * dynamics_cost

    # ---------- Role 3: safe_contact_encouragement ----------
    both_legs_contact = left_contact * right_contact
    x_tol = 0.3
    y_tol = 0.3
    v_tol = 0.2
    angle_tol = 0.1
    angvel_tol = 0.1
    factor_x = max(0.0, 1.0 - abs(x_err) / x_tol)
    factor_y = max(0.0, 1.0 - abs(y_err) / y_tol)
    factor_vx = max(0.0, 1.0 - abs(vx) / v_tol)
    factor_vy = max(0.0, 1.0 - abs(vy) / v_tol)
    factor_angle = max(0.0, 1.0 - abs(angle) / angle_tol)
    factor_angvel = max(0.0, 1.0 - abs(ang_vel) / angvel_tol)
    mean_safety = (factor_x + factor_y + factor_vx + factor_vy + factor_angle + factor_angvel) / 6.0
    safe_contact_bonus = both_legs_contact * mean_safety * 2.0

    total_reward = proximity_reward + soft_landing_penalty + safe_contact_bonus

    components = {
        "proximity_reward": proximity_reward,
        "soft_landing_penalty": soft_landing_penalty,
        "safe_contact_bonus": safe_contact_bonus
    }
    return float(total_reward), components