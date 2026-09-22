def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------- unpack observations --------------------
    # obs layout: [x, y, vx, vy, angle, av, left_contact, right_contact]
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

    # -------------------- weights --------------------
    w_progress = 2.0
    w_vel = 2.0
    w_angvel = 0.5
    w_angle = 0.5
    w_contact = 0.5
    k_proximity = 3.0   # controls how fast velocity penalty ramps up near the pad

    # -------------------- 1. progress reward (improvement_delta) --------------------
    progress_reward = w_progress * (dist - next_dist)

    # -------------------- 2. velocity penalty (soft, proximity‑gated) --------------------
    proximity_gate = 1.0 / (1.0 + k_proximity * next_dist)
    speed_sq = nvx**2 + nvy**2
    vel_penalty = -w_vel * speed_sq * proximity_gate
    ang_penalty = -w_angvel * (n_angvel**2) * proximity_gate

    # -------------------- 3. upright orientation incentive --------------------
    orientation_penalty = -w_angle * (n_angle**2)

    # -------------------- 4. contact reward --------------------
    contact_reward = w_contact * (n_left_contact + n_right_contact)

    # -------------------- total reward --------------------
    total_reward = (progress_reward +
                    vel_penalty +
                    ang_penalty +
                    orientation_penalty +
                    contact_reward)

    components = {
        "progress": progress_reward,
        "velocity_penalty": (vel_penalty + ang_penalty),
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward
    }

    return float(total_reward), components