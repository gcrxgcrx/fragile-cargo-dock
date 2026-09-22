def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current observation
    x_cur, y_cur = obs[0], obs[1]
    xv_cur, yv_cur = obs[2], obs[3]
    angle_cur = obs[4]
    angv_cur = obs[5]
    lc_cur, rc_cur = obs[6], obs[7]

    # Unpack next observation
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    xv_nxt, yv_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angv_nxt = next_obs[5]
    lc_nxt, rc_nxt = next_obs[6], next_obs[7]

    # ---- Potential function ----
    def potential(x, y, xv, yv, angle, angv, lc, rc):
        dist = (x**2 + y**2) ** 0.5
        pos_factor = 1.0 / (1.0 + dist)
        speed = (xv**2 + yv**2 + 0.1 * angv**2) ** 0.5
        speed_factor = 1.0 / (1.0 + speed)
        angle_factor = max(0.0, 1.0 - abs(angle) / 3.14159265)
        contact_factor = (lc + rc) / 2.0   # [0,1]
        return pos_factor * speed_factor * angle_factor * contact_factor

    pot_curr = potential(x_cur, y_cur, xv_cur, yv_cur, angle_cur, angv_cur, lc_cur, rc_cur)
    pot_next = potential(x_nxt, y_nxt, xv_nxt, yv_nxt, angle_nxt, angv_nxt, lc_nxt, rc_nxt)

    potential_gain = 2.0 * (pot_next - pot_curr)

    # ---- Component B: Safe landing penalty (unchanged) ----
    y_pos = next_obs[1]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]

    height_gate = max(0.0, 1.0 - y_pos / 2.0) if y_pos < 2.0 else 0.0
    contact_gate = 0.2 * (left_contact + right_contact)
    landing_gate = height_gate + contact_gate

    motion_cost = x_vel**2 + y_vel**2 + body_angle**2 + 0.1 * ang_vel**2
    safe_landing_penalty = -0.01 * landing_gate * motion_cost

    # ---- Total reward ----
    total_reward = potential_gain + safe_landing_penalty

    components = {
        "potential_gain": potential_gain,
        "safe_landing_penalty": safe_landing_penalty
    }

    return float(total_reward), components