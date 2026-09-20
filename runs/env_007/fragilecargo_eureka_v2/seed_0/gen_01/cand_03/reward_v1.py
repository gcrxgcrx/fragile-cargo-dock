def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    def _sqrt(x):
        return x ** 0.5

    def _atan2(y, x):
        if x == 0.0 and y == 0.0:
            return 0.0
        ax = x if x >= 0.0 else -x
        ay = y if y >= 0.0 else -y
        if ax >= ay:
            t = y / x if x != 0.0 else 0.0
            t2 = t * t
            at = t * (1.0 - t2 / 3.0 + t2 * t2 / 5.0 - t2 * t2 * t2 / 7.0
                      + t2 * t2 * t2 * t2 / 9.0 - t2 * t2 * t2 * t2 * t2 / 11.0)
            ang = at if x > 0.0 else (at + 3.141592653589793 if y >= 0.0 else at - 3.141592653589793)
        else:
            t = x / y if y != 0.0 else 0.0
            t2 = t * t
            at = t * (1.0 - t2 / 3.0 + t2 * t2 / 5.0 - t2 * t2 * t2 / 7.0
                      + t2 * t2 * t2 * t2 / 9.0 - t2 * t2 * t2 * t2 * t2 / 11.0)
            base = 1.5707963267948966 - at
            ang = base if y > 0.0 else base - 3.141592653589793
        return ang

    def _cos(a):
        two_pi = 6.283185307179586
        x = a - two_pi * int(a / two_pi)
        if x > 3.141592653589793:
            x = x - two_pi
        if x < -3.141592653589793:
            x = x + two_pi
        x2 = x * x
        return 1.0 - x2 / 2.0 + x2 * x2 / 24.0 - x2 * x2 * x2 / 720.0 + x2 * x2 * x2 * x2 / 40320.0

    def _clamp(v, lo, hi):
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    dx = obs[12] * 5.0
    dy = obs[13] * 4.0
    ndx = next_obs[12] * 5.0
    ndy = next_obs[13] * 4.0

    dist = _sqrt(dx * dx + dy * dy)
    ndist = _sqrt(ndx * ndx + ndy * ndy)

    cvx = obs[8] * 3.0
    cvy = obs[9] * 3.0
    crate_speed = _sqrt(cvx * cvx + cvy * cvy)

    ncvx = next_obs[8] * 3.0
    ncvy = next_obs[9] * 3.0
    ncrate_speed = _sqrt(ncvx * ncvx + ncvy * ncvy)

    crate_ang = _atan2(obs[11], obs[10])
    align_cos = _cos(crate_ang)
    align = align_cos if align_cos >= 0.0 else -align_cos

    # 1) main progress: delta distance, convexified to break low-level plateau
    progress = _clamp(dist - ndist, -1.0, 1.0)
    if progress >= 0.0:
        c_progress = 14.0 * progress + 8.0 * progress * progress
    else:
        c_progress = 14.0 * progress

    # 2) settling: only meaningful when crate is genuinely near dock AND
    #    the crate is not being driven away (progress >= 0). Gated product,
    #    not a standalone global reward.
    near_gate = _clamp(1.0 - dist / 1.2, 0.0, 1.0)
    speed_factor = 1.0 / (1.0 + 25.0 * crate_speed)
    c_settle = 2.0 * near_gate * speed_factor

    # 3) alignment: gated by nearness, product form so it cannot be farmed
    #    from far away.
    c_align = 1.2 * near_gate * align

    # 4) joint completion proxy: tight near + slow + aligned, geometric mean.
    #    Strongly downweighted vs previous version to avoid dominating.
    f_near = _clamp(1.0 - dist / 0.8, 0.0, 1.0)
    f_slow = 1.0 / (1.0 + 40.0 * crate_speed)
    f_align = align
    joint = (f_near * f_slow * f_align) ** (1.0 / 3.0)
    c_joint = 1.5 * joint

    # 5) gentle contact: penalize high relative speed while in contact
    contact = obs[14]
    cart_fwd = obs[4] * 3.0
    rel_speed = _sqrt((cart_fwd - cvx) * (cart_fwd - cvx) + cvy * cvy)
    hard_risk = _clamp(rel_speed - 0.6, 0.0, 3.0)
    c_gentle = -1.5 * contact * hard_risk

    # 6) boundary avoidance: hinge penalty near warehouse edges
    cart_x = obs[0] * 5.0
    cart_y = obs[1] * 4.0
    cc = obs[2]
    ss = obs[3]
    relx = obs[6] * 3.0
    rely = obs[7] * 3.0
    crate_x = cart_x + cc * relx - ss * rely
    crate_y = cart_y + ss * relx + cc * rely
    crate_margin = 0.6
    cart_margin = 0.6
    c_bound = 0.0
    c_bound -= 3.0 * _clamp(crate_margin - (5.0 - abs(crate_x)), 0.0, 5.0)
    c_bound -= 3.0 * _clamp(crate_margin - (4.0 - abs(crate_y)), 0.0, 4.0)
    c_bound -= 2.0 * _clamp(cart_margin - (5.0 - abs(cart_x)), 0.0, 5.0)
    c_bound -= 2.0 * _clamp(cart_margin - (4.0 - abs(cart_y)), 0.0, 4.0)

    # 7) wall proximity light penalty
    wall_near = obs[15] if obs[15] > obs[16] else obs[16]
    if obs[17] > wall_near:
        wall_near = obs[17]
    c_wall = -0.4 * _clamp(wall_near - 0.85, 0.0, 1.0)

    components = {
        "crate_to_dock_progress": c_progress,
        "crate_settling": c_settle,
        "crate_dock_alignment": c_align,
        "joint_completion_proxy": c_joint,
        "gentle_contact": c_gentle,
        "boundary_avoidance": c_bound,
        "wall_proximity": c_wall,
    }

    total = 0.0
    for k in components:
        total += components[k]

    return float(total), components