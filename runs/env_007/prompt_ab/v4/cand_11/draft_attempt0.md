```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode boundary detection ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    components = {}

    # ---------- geometry / signals ----------
    # crate-to-dock offsets (normalized by half width/height)
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dx_old = float(obs[12])
    dy_old = float(obs[13])

    dist = (dx * dx + dy * dy) ** 0.5
    dist_old = (dx_old * dx_old + dy_old * dy_old) ** 0.5

    # crate speed (m/s)
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # crate heading error relative to dock (dock aligned with +x assumed)
    cos_h = float(next_obs[10])
    sin_h = float(next_obs[11])
    heading_err = (sin_h * sin_h) ** 0.5  # |sin| as a proxy for angle error
    # more robust: angle error magnitude via |sin| (0 aligned, 1 perpendicular)
    align = 1.0 - heading_err  # 1 aligned, 0 perpendicular

    # contact + closing speed (gentleness proxy)
    crate_vx = cvx
    crate_vy = cvy
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0

    # completion condition (from environment facts)
    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0
    aligned = 1.0 if heading_err < 0.5 else 0.0   # <30deg -> |sin|<0.5
    slow = 1.0 if crate_speed < 0.05 else 0.0
    complete_now = 1.0 if (inside > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0

    if complete_now > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- success event (one-time, module-level) ----------
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # first-entry one-time bonus
    entered_bonus = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        entered_bonus = 20.0

    # ---------- process components (must be 0 in completed state) ----------
    # main: incremental progress toward dock (only rewards getting closer)
    progress = dist_old - dist
    if progress < 0.0:
        progress = 0.0
    # gate by alignment so pushing straight counts, sideways pushing less
    progress_gate = 0.5 + 0.5 * align
    crate_progress = 12.0 * progress * progress_gate

    # near-dock speed suppression: only active when outside dock tolerance
    # (hinge: penalize high crate speed when close to dock but not inside)
    near_dock = 1.0 if (dist < 0.15 and inside < 0.5) else 0.0
    speed_excess = crate_speed - 0.05
    if speed_excess < 0.0:
        speed_excess = 0.0
    speed_penalty = -8.0 * near_dock * speed_excess

    # gentleness on contact (only penalize closing while in contact)
    gentleness = -0.05 * contact * closing

    # obstacle proximity hinge penalty (front sensor)
    front = float(next_obs[15])
    obs_excess = front - 0.7
    if obs_excess < 0.0:
        obs_excess = 0.0
    obstacle_penalty = -2.0 * obs_excess

    # out-of-bounds hinge on cart position (normalized to ~[-1,1])
    cx = float(next_obs[0])
    cy = float(next_obs[1])
    oob = 0.0
    ax = abs(cx)
    ay = abs(cy)
    if ax > 0.9:
        oob += (ax - 0.9)
    if ay > 0.9:
        oob += (ay - 0.9)
    oob_penalty = -10.0 * oob

    # ---------- zero-out process terms in completed state ----------
    if complete_now > 0.5:
        crate_progress = 0.0
        speed_penalty = 0.0
        gentleness = 0.0
        obstacle_penalty = 0.0
        oob_penalty = 0.0

    components["crate_progress"] = crate_progress
    components["speed_penalty"] = speed_penalty
    components["gentleness"] = gentleness
    components["obstacle_penalty"] = obstacle_penalty
    components["oob_penalty"] = oob_penalty
    components["entered_bonus"] = entered_bonus
    components["success_event"] = success_event

    total = (crate_progress + speed_penalty + gentleness + obstacle_penalty
             + oob_penalty + entered_bonus + success_event)

    return float(total), components
```