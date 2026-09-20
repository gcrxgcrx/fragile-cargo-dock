```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- module-level state (containers, no global statement) ----
    _PREV_T = [-1.0]
    _STREAK = [0]
    _PAID = [False]
    _ENTERED = [False]

    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---- geometry / signal extraction (from declared obs indices) ----
    # crate-to-dock signed offsets, normalized
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    odx = float(obs[12])
    ody = float(obs[13])

    # distance to dock center (normalized units, combine both axes)
    dist = (dx * dx + dy * dy) ** 0.5
    odist = (odx * odx + ody * ody) ** 0.5

    # crate world velocity magnitude (m/s)
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # crate heading error vs dock (aligned along +x assumed; use crate heading)
    # crate heading angle
    # orientation alignment: use cos of heading relative to world axes
    ch = float(next_obs[10])  # crate_cos_heading
    sh = float(next_obs[11])  # crate_sin_heading
    # alignment quality: |cos| close to 1 means aligned with world x
    align = 1.0 - abs(sh)  # in [0,1], 1 = perfectly axis-aligned

    # ---- component 1: crate_to_dock_progress (incremental) ----
    # only reward actual reduction of distance to dock center
    progress = odist - dist
    if progress < 0.0:
        progress = progress  # keep negative (moving away penalized mildly)
    crate_progress = 6.0 * progress

    # ---- component 2: docking quality shaping (only near dock) ----
    # gated by being close to dock
    near_gate = 1.0 / (1.0 + 8.0 * dist)
    # alignment term (continuous, only near dock)
    align_term = 0.6 * near_gate * align
    # low crate speed near dock
    speed_near = 0.0
    if dist < 0.15:
        speed_near = -0.8 * (crate_speed ** 2)

    # ---- component 3: gentleness (soft contact) ----
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---- component 4: out_of_bounds hinge penalty ----
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    oob = 0.0
    # cart near boundary (normalized, bounds ~[-1,1])
    if abs(cart_x) > 0.85:
        oob -= 0.5 * (abs(cart_x) - 0.85)
    if abs(cart_y) > 0.85:
        oob -= 0.5 * (abs(cart_y) - 0.85)

    # ---- component 5: obstacle proximity penalty (front sensor) ----
    front = float(next_obs[15])
    obstacle_pen = 0.0
    if front > 0.6:
        obstacle_pen = -0.3 * (front - 0.6)

    # ---- completion condition (explicit from obs, tight tolerance) ----
    # |dx| <= 0.024 and |dy| <= 0.030 (from environment card)
    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0
    aligned = 1.0 if abs(sh) < 0.5 else 0.0  # heading error < 30 deg
    still = 1.0 if crate_speed < 0.05 else 0.0

    complete_now = 1.0 if (inside > 0.5 and aligned > 0.5 and still > 0.5) else 0.0

    if complete_now > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # one-time first-entry bonus
    enter_bonus = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_bonus = 15.0

    # one-time success event
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    total = (
        crate_progress
        + align_term
        + speed_near
        + gentleness
        + oob
        + obstacle_pen
        + enter_bonus
        + success_event
    )

    components = {
        "crate_progress": crate_progress,
        "align_term": align_term,
        "speed_near": speed_near,
        "gentleness": gentleness,
        "oob": oob,
        "obstacle_pen": obstacle_pen,
        "enter_bonus": enter_bonus,
        "success_event": success_event,
    }

    return (float(total), components)
```