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

    # ---------- geometry / signals ----------
    # crate-to-dock signed offsets (normalized by half-width / half-height)
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dx0 = float(obs[12])
    dy0 = float(obs[13])

    # dock tolerance (from env card): |dx| <= 0.024, |dy| <= 0.030
    inside_x = 1.0 if abs(dx) <= 0.024 else 0.0
    inside_y = 1.0 if abs(dy) <= 0.030 else 0.0
    inside = 1.0 if (inside_x > 0.5 and inside_y > 0.5) else 0.0

    # crate orientation error (radians)
    crate_ang = 0.0
    ccos = float(next_obs[10])
    csin = float(next_obs[11])
    # atan2 via approximation is not allowed; use cos of angle error directly.
    # crate heading vs dock heading: dock heading assumed aligned with world +x
    # so alignment = |cos(theta_crate)| ; error < 30deg => cos >= cos(30)=0.866
    align_cos = ccos if ccos > 0.0 else -ccos
    aligned = 1.0 if align_cos >= 0.866 else 0.0

    # crate speed (m/s)
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    slow = 1.0 if crate_speed < 0.05 else 0.0

    # ---------- completion condition & streak ----------
    if inside > 0.5 and aligned > 0.5 and slow > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    completed = 1.0 if (_STREAK[0] >= 10) else 0.0

    # ---------- components ----------
    components = {}

    # === 1. crate-to-dock progress (incremental, gated) ===
    # distance measure in normalized units
    d0 = (dx0 * dx0 + dy0 * dy0) ** 0.5
    d1 = (dx * dx + dy * dy) ** 0.5
    progress = d0 - d1  # positive when crate moved closer
    if progress < 0.0:
        progress = 0.0
    # gate: only reward progress when not yet completed
    not_done = 1.0 - completed
    progress_reward = 6.0 * progress * not_done
    components["crate_progress"] = progress_reward

    # === 2. contact gentleness (soft contact penalty) ===
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -3.0 * contact * closing
    components["soft_contact"] = gentleness

    # === 3. speed penalty near dock (hinge, only when close) ===
    near_dock = 1.0 if (abs(dx) < 0.15 and abs(dy) < 0.18) else 0.0
    speed_excess = crate_speed - 0.05
    if speed_excess < 0.0:
        speed_excess = 0.0
    speed_penalty = -8.0 * near_dock * speed_excess
    components["dock_speed_penalty"] = speed_penalty

    # === 4. out-of-bounds guard (cart + crate) ===
    cx = abs(float(next_obs[0]))
    cy = abs(float(next_obs[1]))
    cart_margin = cx if cx > cy else cy
    cart_oob = cart_margin - 0.95
    if cart_oob < 0.0:
        cart_oob = 0.0
    # crate world position recovery for boundary guard
    rel_x = float(next_obs[6]) * 3.0
    rel_y = float(next_obs[7]) * 3.0
    ch = float(next_obs[2])
    sh = float(next_obs[3])
    crate_wx = float(next_obs[0]) * 5.0 + rel_x * ch - rel_y * sh
    crate_wy = float(next_obs[1]) * 4.0 + rel_x * sh + rel_y * ch
    # normalize crate world pos roughly by same half-extents
    crate_nx = abs(crate_wx) / 5.0
    crate_ny = abs(crate_wy) / 4.0
    crate_margin = crate_nx if crate_nx > crate_ny else crate_ny
    crate_oob = crate_margin - 0.95
    if crate_oob < 0.0:
        crate_oob = 0.0
    oob_penalty = -20.0 * (cart_oob + crate_oob)
    components["out_of_bounds"] = oob_penalty

    # === 5. one-time "entered dock" bonus ===
    entered_bonus = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        entered_bonus = 15.0
    components["entered_dock_bonus"] = entered_bonus

    # === 6. one-time success event ===
    success_event = 0.0
    if completed > 0.5 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["success_event"] = success_event

    # ---------- zero-out all non-event components when completed ----------
    if completed > 0.5:
        components["crate_progress"] = 0.0
        components["soft_contact"] = 0.0
        components["dock_speed_penalty"] = 0.0
        components["out_of_bounds"] = 0.0
        # entered_dock_bonus already one-time; keep as-is (it's 0 after first fire)
        # but must be exactly 0 when in completed steady state (already fired)
        if _ENTERED[0]:
            components["entered_dock_bonus"] = 0.0

    total = (
        components["crate_progress"]
        + components["soft_contact"]
        + components["dock_speed_penalty"]
        + components["out_of_bounds"]
        + components["entered_dock_bonus"]
        + components["success_event"]
    )

    return (float(total), components)
```