def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next observation
    x = next_obs[0]           # horizontal position (relative to target)
    y = next_obs[1]           # vertical position (relative to pad)
    vx = next_obs[2]          # horizontal velocity
    vy = next_obs[3]          # vertical velocity
    angle = next_obs[4]       # body angle
    angvel = next_obs[5]      # angular velocity
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Derived quantities
    dist = (x**2 + y**2)**0.5
    speed = (vx**2 + vy**2)**0.5

    # Exponential helpers (no imports allowed)
    exp_neg_dist = 2.718281828 ** (-dist)
    exp_neg_speed = 2.718281828 ** (-speed)

    # 1. Goal proximity – main learning signal (bounded positive reward)
    goal_proximity = exp_neg_dist

    # 2. Safe contact soft proxy – joint condition: both contacts and low speed
    contact_proxy = left_contact * right_contact * exp_neg_speed

    # 3. Velocity damping – distance‑gated speed penalty
    velocity_damping = -0.2 * speed * exp_neg_dist

    # 4. Orientation stability – quadratic penalties on angle and angular velocity
    angle_penalty = -0.5 * (angle ** 2)
    angvel_penalty = -0.1 * (angvel ** 2)

    # Aggregate reward
    total = goal_proximity + contact_proxy + velocity_damping + angle_penalty + angvel_penalty

    components = {
        'goal_proximity': goal_proximity,
        'safe_contact_proxy': contact_proxy,
        'velocity_damping': velocity_damping,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty
    }

    return float(total), components