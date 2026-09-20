```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_DIST = [-1.0]

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode boundary detection ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = -1.0
    _PREV_T[0] = t

    # ---------- geometry ----------
    # dock offsets (normalized by half-width / half-height)
    off_x = float(next_obs[12])
    off_y = float(next_obs[13])
    dist = (off_x * off_x + off_y * off_y) ** 0.5

    # completion conditions (from environment facts)
    inside = 1.0 if (abs(off_x) <= 0.024 and abs(off_y) <= 0.030) else 0.0

    # crate heading error vs dock (crate_cos_heading, crate_sin_heading)
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    # crate heading angle
    # dock alignment: crate heading should match +x dock axis -> cos ~ 1
    align = ch  # cos of heading; 1.0 = aligned with +x
    aligned = 1.0 if align >= 0.8660254 else 0.0  # cos(30 deg)

    # crate speed (world), scaled by 3.0 m/s
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    slow = 1.0 if crate_speed < 0.05 else 0.0

    # ---------- completion streak ----------
    done_cond = (inside > 0.5) and (aligned > 0.5) and (slow > 0.5)
    if done_cond:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- contact / gentleness proxy ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    chx = float(obs[2])
    chy = float(obs[3])
    crate_along_heading = crate_vx * chx + crate_vy * chy
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---------- boundary guards ----------
    cx = float(next_obs[0])
    cy = float(next_obs[1])
    # crate world position recovery
    cart_x = cx * 5.0
    cart_y = cy * 4.0
    rel_x = float(next_obs[6]) * 3.0
    rel_y = float(next_obs[7]) * 3.0
    crate_wx = cart_x + rel_x * chx - rel_y * chy
    crate_wy = cart_y + rel_x * chy + rel_y * chx
    crate_nx = crate_wx / 5.0
    crate_ny = crate_wy / 4.0

    guard = 0.0
    # cart boundary
    ax = abs(cx)
    ay = abs(cy)
    if ax > 0.95:
        guard -= 6.0 * (ax - 0.95) * (ax - 0.95) * 100.0
    if ay > 0.95:
        guard -= 6.0 * (ay - 0.95) * (ay - 0.95) * 100.0
    # crate boundary
    acx = abs(crate_nx)
    acy = abs(crate_ny)
    if acx > 0.95:
        guard -= 6.0 * (acx - 0.95) * (acx - 0.95) * 100.0
    if acy > 0.95:
        guard -= 6.0 * (acy - 0.95) * (acy - 0.95) * 100.0
    if guard < -50.0:
        guard = -50.0

    # ---------- progress (improvement delta on dock distance) ----------
    if _PREV_DIST[0] < 0.0:
        # first step of episode: no delta
        delta = 0.0
    else:
        delta = _PREV_DIST[0] - dist
    _PREV_DIST[0] = dist

    # scale delta: normalized distance units. Full field ~ 1.0
    progress = 0.0
    if delta > 0.0:
        progress = 40.0 * delta
    else:
        progress = 25.0 * delta  # negative, penalize moving away (smaller magnitude)

    # ---------- success / entry one-time events ----------
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    entry_event = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        entry_event = 20.0

    # ---------- freeze all non-event components when in completion state ----------
    in_completion_state = (inside > 0.5) and (aligned > 0.5) and (slow > 0.5)
    if in_completion_state:
        progress = 0.0
        gentleness = 0.0
        guard = 0.0

    total = progress + gentleness + guard + success_event + entry_event

    components = {
        "progress": progress,
        "gentleness": gentleness,
        "boundary_guard": guard,
        "success_event": success_event,
        "entry_event": entry_event,
    }
    return float(total), components
```