_PREV_T = [-1.0]
_PREV_DIST = [-1.0]

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode boundary detection ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _PREV_DIST[0] = -1.0
    _PREV_T[0] = t

    # ---------- geometry from observations ----------
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    # crate heading error vs dock (aligned to +x axis): |sin(theta)|
    sh = float(next_obs[11])
    heading_err = (sh * sh) ** 0.5

    # crate speed (m/s)
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---------- continuous completion-proxy factors (all in [0,1]) ----------
    # position factor: 1 when inside dock, decays with distance
    pos_f = 1.0 / (1.0 + 12.0 * dist)
    # alignment factor: 1 when aligned, 0 when perpendicular
    align_f = 1.0 - heading_err
    if align_f < 0.0:
        align_f = 0.0
    # stillness factor: 1 when crate nearly stopped
    still_f = 1.0 / (1.0 + 20.0 * crate_speed)

    components = {}

    # ---------- main dense potential: distance to dock (positive, monotone) ----------
    # potential Phi = 1/(1+dist); reward on improvement of Phi
    phi_now = 1.0 / (1.0 + dist)
    phi_prev = 1.0 / (1.0 + _PREV_DIST[0]) if _PREV_DIST[0] >= 0.0 else phi_now
    _PREV_DIST[0] = dist

    # improvement in potential (>=0 when approaching, negative when retreating)
    dphi = phi_now - phi_prev

    # gate the potential-improvement by alignment & stillness so that near-dock
    # the agent is nudged toward the full completion configuration.
    # gate is a *multiplier on progress*, never a standalone global reward.
    gate = 0.35 + 0.65 * (0.5 * align_f + 0.5 * still_f)
    if gate < 0.0:
        gate = 0.0
    if gate > 1.0:
        gate = 1.0

    progress = 120.0 * dphi * gate

    # ---------- joint soft-completion proxy (geometric mean of factors) ----------
    # continuous, always gives gradient; peaks when inside+aligned+still.
    joint = (pos_f * align_f * still_f) ** (1.0 / 3.0)
    completion = 40.0 * joint

    # ---------- gentleness: penalize hard closing on contact ----------
    crate_along = cvx * float(obs[2]) + cvy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -1.5 * contact * closing

    # ---------- speed hinge near dock (only when close, only over-threshold) ----------
    speed_gate_penalty = 0.0
    if dist < 0.25 and crate_speed > 0.06:
        over = crate_speed - 0.06
        speed_gate_penalty = -6.0 * over * over

    # ---------- out-of-bounds hinge (cart only; crate via dock offset) ----------
    cx = float(next_obs[0])
    cy = float(next_obs[1])
    ax = cx if cx >= 0.0 else -cx
    ay = cy if cy >= 0.0 else -cy
    bounds_penalty = 0.0
    if ax > 0.92:
        bounds_penalty -= 30.0 * (ax - 0.92)
    if ay > 0.92:
        bounds_penalty -= 30.0 * (ay - 0.92)
    # crate far from dock beyond plausible region -> mild push back
    if dist > 1.4:
        bounds_penalty -= 4.0 * (dist - 1.4)

    # ---------- action smoothness (very light; must not suppress pushing) ----------
    a0 = float(action[0])
    a1 = float(action[1])
    action_penalty = -0.003 * (a0 * a0 + a1 * a1)

    components["progress"] = progress
    components["completion"] = completion
    components["gentleness"] = gentleness
    components["speed_gate_penalty"] = speed_gate_penalty
    components["bounds_penalty"] = bounds_penalty
    components["action_penalty"] = action_penalty

    total = (progress + completion + gentleness
             + speed_gate_penalty + bounds_penalty + action_penalty)
    return (float(total), components)