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
    speed = (vx**2 + vy**2)**0.5
    dist = (x**2 + y**2)**0.5

    # 1. Decomposed proximity
    vertical_proximity = 2.718281828 ** (-abs(y))          # [0,1]
    horizontal_alignment = 2.718281828 ** (- (x**2))       # [0,1]
    goal_proximity = vertical_proximity * horizontal_alignment

    # 2. Landing success – geometric mean for non-collapsing gradient
    contact_score = (left_contact + right_contact) / 2.0   # [0, 0.5, 1]
    near_target = max(0.0, 1.0 - dist / 1.5)               # [0,1], hinge at 1.5m
    landing_success = 2.0 * (contact_score * near_target) ** 0.5  # geometric mean prevents collapse to 0

    # 3. Velocity damping – reduced coefficient
    velocity_damping = -0.05 * speed * (2.718281828 ** (-dist))

    # 4. Orientation stability
    angle_penalty = -0.1 * (angle ** 2)
    angvel_penalty = -0.1 * (angvel ** 2)

    # Aggregate reward
    total_reward = goal_proximity + landing_success + velocity_damping + angle_penalty + angvel_penalty

    components = {
        'goal_proximity': goal_proximity,
        'landing_success': landing_success,
        'velocity_damping': velocity_damping,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty
    }

    return float(total_reward), components