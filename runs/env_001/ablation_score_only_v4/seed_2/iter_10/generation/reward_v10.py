def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------- unpack observations --------------------
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # -------------------- distance to target pad --------------------
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # -------------------- derived quantities --------------------
    speed = (nvx**2 + nvy**2) ** 0.5
    abs_angle = abs(n_angle)

    # -------------------- weights --------------------
    w_progress = 2.0
    w_vel = 0.2
    w_angvel = 0.1
    w_angle = 0.15
    w_landing = 1.0          # reduced to avoid dominating
    k_speed = 2.0
    k_angle = 2.0

    # -------------------- 1. progress reward --------------------
    progress_reward = w_progress * (dist - next_dist)

    # -------------------- 2. velocity penalty (local gate) --------------------
    local_gate = max(0.0, 1.0 - next_dist)
    speed_sq = nvx**2 + nvy**2
    vel_penalty = -w_vel * speed_sq * local_gate
    ang_penalty = -w_angvel * (n_angvel**2) * local_gate

    # -------------------- 3. upright orientation incentive (gated) --------------------
    orientation_penalty = -w_angle * (abs_angle**2) * local_gate

    # -------------------- 4. safe landing reward (gated by safety) --------------------
    contact_sum = n_left_contact + n_right_contact

    # Safety gate: only reward landing if both speed and angle are within safe limits
    safe_speed = 0.5
    safe_angle = 0.2
    if speed <= safe_speed and abs_angle <= safe_angle:
        speed_bonus = max(0.0, 1.0 - k_speed * speed)
        angle_bonus = max(0.0, 1.0 - k_angle * abs_angle)
        landing_factor = 1.0 + speed_bonus + angle_bonus
    else:
        landing_factor = 0.0   # unsafe landing receives no reward

    landing_reward = w_landing * contact_sum * landing_factor

    # -------------------- total reward --------------------
    total_reward = (progress_reward +
                    vel_penalty +
                    ang_penalty +
                    orientation_penalty +
                    landing_reward)

    components = {
        "progress": progress_reward,
        "velocity_penalty": (vel_penalty + ang_penalty),
        "orientation_penalty": orientation_penalty,
        "landing_reward": landing_reward
    }

    return float(total_reward), components