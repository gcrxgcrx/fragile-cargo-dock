def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- unpack ----------
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

    cart_x = obs[0]
    cart_y = obs[1]

    # ---------- A. joint_dock_completion (MAIN, geometric mean, non-collapsing) ----------
    # factor 1: near dock (continuous, 1 at center, 0 at radius 0.6)
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    f_near = max(0.0, 1.0 - dist_new / 0.6)

    # factor 2: slow (1 at rest, decays with speed)
    f_slow = 1.0 / (1.0 + 30.0 * crate_speed)

    # factor 3: aligned (|cos| of crate heading vs dock axis; dock axis assumed x)
    f_align = abs(crate_cos)

    # geometric mean with floors to avoid total collapse
    f_near_c = max(f_near, 0.05)
    f_slow_c = max(f_slow, 0.05)
    f_align_c = max(f_align, 0.05)
    joint = (f_near_c * f_slow_c * f_align_c) ** (1.0 / 3.0)

    w_joint = 6.0
    r_joint = w_joint * joint

    # ---------- B. crate_to_dock_progress (auxiliary improvement delta) ----------
    dist_old = (dx * dx + dy * dy) ** 0.5
    progress = dist_old - dist_new
    w_progress = 15.0
    r_progress = w_progress * progress

    # ---------- C. fragile_impact_penalty (hinge, only on contact) ----------
    rel_speed = abs(cart_fwd - crate_speed)
    impact_excess = max(0.0, rel_speed - 0.15)
    w_impact = 4.0
    r_impact = -w_impact * impact_excess * contact

    # ---------- D. boundary_health_penalty (hinge, replaces zombie) ----------
    bx = max(0.0, abs(cart_x) - 0.9)
    by = max(0.0, abs(cart_y) - 0.9)
    w_bounds = 6.0
    r_bounds = -w_bounds * (bx + by)

    # ---------- total ----------
    total_reward = r_joint + r_progress + r_impact + r_bounds

    components = {
        "joint_dock_completion": r_joint,
        "crate_to_dock_progress": r_progress,
        "fragile_impact_penalty": r_impact,
        "boundary_health_penalty": r_bounds,
    }

    return float(total_reward), components