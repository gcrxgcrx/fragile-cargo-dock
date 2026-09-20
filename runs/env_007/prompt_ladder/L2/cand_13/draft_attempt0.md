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

    # ---------- geometry ----------
    # crate-to-dock offsets (normalized)
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    odx = float(obs[12])
    ody = float(obs[13])
    old_dist = (odx * odx + ody * ody) ** 0.5

    # crate heading error to dock (crate heading is the "alignment" reference)
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    # dock reference heading: assume +x axis; angle error via cos of crate heading
    align = crate_cos  # in [-1, 1], 1 = aligned with +x
    if align < 0.0:
        align = 0.0

    # crate speed (m/s)
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---------- completion condition (explicit from obs) ----------
    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0
    aligned = 1.0 if crate_cos >= 0.8660254 else 0.0   # within 30 deg
    slow = 1.0 if crate_speed < 0.05 else 0.0
    complete_now = 1.0 if (inside > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0

    if complete_now > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- components ----------
    components = {}

    # 1) progress: incremental crate-to-dock improvement (only when closer)
    progress = old_dist - dist
    if progress < 0.0:
        progress = 0.0
    # gate by alignment (soft) so pushing in the right direction is rewarded
    align_gate = 0.3 + 0.7 * align
    progress_reward = 12.0 * progress * align_gate
    components["crate_to_dock_progress"] = progress_reward

    # 2) soft contact gentleness (closing speed proxy)
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.8 * contact * closing
    components["soft_contact_gentleness"] = gentleness

    # 3) near-dock speed penalty (hinge: only when close to dock and fast)
    near = 1.0 - min(1.0, dist / 0.15)
    if near < 0.0:
        near = 0.0
    excess_speed = crate_speed - 0.05
    if excess_speed < 0.0:
        excess_speed = 0.0
    speed_penalty = -3.0 * near * excess_speed
    components["crate_speed_near_dock"] = speed_penalty

    # 4) out-of-bounds guard for cart (monotone penalty approaching wall)
    cart_x = abs(float(next_obs[0]))
    cart_y = abs(float(next_obs[1]))
    oob = 0.0
    if cart_x > 0.95:
        oob += (cart_x - 0.95)
    if cart_y > 0.95:
        oob += (cart_y - 0.95)
    oob_penalty = -60.0 * oob
    components["cart_out_of_bounds"] = oob_penalty

    # 5) crate out-of-bounds guard (recover crate world position)
    # cart world pos (approx, using warehouse half-width 5.0, half-height 4.0)
    cart_wx = float(next_obs[0]) * 5.0
    cart_wy = float(next_obs[1]) * 4.0
    ch = float(next_obs[2])
    sh = float(next_obs[3])
    rel_x = float(next_obs[6]) * 3.0
    rel_y = float(next_obs[7]) * 3.0
    crate_wx = cart_wx + rel_x * ch - rel_y * sh
    crate_wy = cart_wy + rel_x * sh + rel_y * ch
    crate_nx = abs(crate_wx) / 5.0
    crate_ny = abs(crate_wy) / 4.0
    crate_oob = 0.0
    if crate_nx > 0.95:
        crate_oob += (crate_nx - 0.95)
    if crate_ny > 0.95:
        crate_oob += (crate_ny - 0.95)
    components["crate_out_of_bounds"] = -60.0 * crate_oob

    # 6) obstacle proximity penalty (avoid ramming walls)
    sf = float(next_obs[15])
    sl = float(next_obs[16])
    sr = float(next_obs[17])
    obst = 0.0
    if sf > 0.7:
        obst += (sf - 0.7)
    if sl > 0.7:
        obst += (sl - 0.7)
    if sr > 0.7:
        obst += (sr - 0.7)
    components["obstacle_proximity"] = -2.0 * obst

    # 7) one-time entry bonus
    entry_bonus = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        entry_bonus = 15.0
    components["dock_entry_bonus"] = entry_bonus

    # 8) one-time success event
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["dock_success_event"] = success_event

    # ---------- zero-out when in completed stable state ----------
    if complete_now > 0.5 and success_event == 0.0 and entry_bonus == 0.0:
        for k in components:
            components[k] = 0.0

    total = 0.0
    for k in components:
        total += components[k]

    return (float(total), components)
```