def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- unpack signals ---
    # crate-to-dock offsets (normalized by half-width 5.0, half-height 4.0)
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    # crate velocity (world frame, normalized by 3.0)
    cvx = obs[8]
    cvy = obs[9]
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # crate heading
    crate_cos = obs[10]
    crate_sin = obs[11]

    # cart forward speed (normalized by 3.0)
    cart_fwd = obs[4]

    # contact flag
    contact = obs[14]

    # --- A. crate_to_dock_progress: improvement_delta on distance ---
    dist_old = (dx * dx + dy * dy) ** 0.5
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    progress = dist_old - dist_new
    w_progress = 10.0
    r_progress = w_progress * progress

    # --- B. crate_settling_and_alignment: joint_condition_proxy ---
    # proximity factor: 1 when at dock, decays with distance
    prox = 1.0 / (1.0 + 8.0 * dist_new)
    # speed factor: 1 when still, decays with speed
    speed_factor = 1.0 / (1.0 + 20.0 * crate_speed)
    # alignment factor: crate heading aligned with dock axis (assume dock axis = world x)
    # |cos(heading)| close to 1 means aligned with x-axis
    align = abs(crate_cos)
    # geometric mean of three continuous factors
    settle = (prox * speed_factor * align) ** (1.0 / 3.0)
    w_settle = 3.0
    r_settle = w_settle * settle

    # --- C. fragile_impact_penalty: hinge on relative speed during contact ---
    # relative speed proxy: cart forward speed vs crate speed
    rel_speed = abs(cart_fwd - crate_speed)
    # hinge: only penalize when contact AND relative speed exceeds threshold
    impact_threshold = 0.15
    impact_excess = max(0.0, rel_speed - impact_threshold)
    w_impact = 5.0
    r_impact = -w_impact * impact_excess * contact

    # --- D. out_of_bounds_penalty: hinge on cart position near boundary ---
    cart_x = obs[0]
    cart_y = obs[1]
    # normalized bounds [-1, 1]; penalize when |pos| > 0.85
    bound_threshold = 0.85
    cart_x_excess = max(0.0, abs(cart_x) - bound_threshold)
    cart_y_excess = max(0.0, abs(cart_y) - bound_threshold)
    w_bounds = 8.0
    r_bounds = -w_bounds * (cart_x_excess + cart_y_excess)

    # --- total ---
    total_reward = r_progress + r_settle + r_impact + r_bounds

    components = {
        "crate_to_dock_progress": r_progress,
        "crate_settling_and_alignment": r_settle,
        "fragile_impact_penalty": r_impact,
        "out_of_bounds_penalty": r_bounds,
    }

    return float(total_reward), components