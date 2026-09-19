def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- unpack signals ---
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    cvx = obs[8]
    cvy = obs[9]
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    crate_cos = obs[10]
    crate_sin = obs[11]

    cart_fwd = obs[4]
    contact = obs[14]

    # --- A. crate_to_dock_progress: improvement_delta on distance (main signal) ---
    dist_old = (dx * dx + dy * dy) ** 0.5
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    progress = dist_old - dist_new
    w_progress = 20.0
    r_progress = w_progress * progress

    # --- B. crate_settling_and_alignment: improvement-gated, not state-value ---
    # Only reward when crate is near dock AND actually converging (progress>0)
    # or already nearly stopped inside the dock region.
    prox_hinge = max(0.0, 1.0 - dist_new / 0.6)
    speed_factor = 1.0 / (1.0 + 20.0 * crate_speed)
    align = abs(crate_cos)
    settle_raw = (prox_hinge * speed_factor * align) ** (1.0 / 3.0)
    # improvement gate: reward only when crate is approaching dock this step
    approach_gate = max(0.0, min(1.0, progress * 20.0))
    # allow a small floor only when already very close and slow (final settling)
    near_still = prox_hinge * speed_factor
    gate = max(approach_gate, 0.3 * near_still)
    w_settle = 1.0
    r_settle = w_settle * settle_raw * gate

    # --- C. fragile_impact_penalty: hinge on relative speed during contact ---
    rel_speed = abs(cart_fwd - crate_speed)
    impact_threshold = 0.15
    impact_excess = max(0.0, rel_speed - impact_threshold)
    w_impact = 5.0
    r_impact = -w_impact * impact_excess * contact

    # --- D. out_of_bounds_penalty: hinge on cart position near boundary ---
    cart_x = obs[0]
    cart_y = obs[1]
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