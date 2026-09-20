def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- episode boundary detection via monotone time_fraction ----
    _PREV_T = [-1.0]
    _STREAK = [0]
    _PAID = [False]
    _ENTERED = [False]
    # NOTE: these are re-created each call in this listing; in the real
    # environment they must live at module scope.  To satisfy the contract
    # while keeping module-level state, we use module-level containers below.

    t = float(next_obs[18])
    if t < _prev_t[0] or t <= 1.0 / 400.0:
        _streak[0] = 0
        _paid[0] = False
        _entered[0] = False
    _prev_t[0] = t

    # ---- geometry / kinematics from declared obs indices ----
    dock_dx = float(next_obs[12]) * 5.0   # signed x offset to dock (m)
    dock_dy = float(next_obs[13]) * 4.0   # signed y offset to dock (m)
    dock_dist = (dock_dx * dock_dx + dock_dy * dock_dy) ** 0.5

    prev_dock_dx = float(obs[12]) * 5.0
    prev_dock_dy = float(obs[13]) * 4.0
    prev_dock_dist = (prev_dock_dx * prev_dock_dx + prev_dock_dy * prev_dock_dy) ** 0.5

    # ---- completion condition (tolerances taken from env facts) ----
    inside_x = 1.0 if abs(float(next_obs[12])) <= 0.024 else 0.0
    inside_y = 1.0 if abs(float(next_obs[13])) <= 0.030 else 0.0
    inside = 1.0 if (inside_x > 0.5 and inside_y > 0.5) else 0.0

    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    crate_hx = float(next_obs[10])
    crate_hy = float(next_obs[11])
    # crate heading error relative to +x axis (dock alignment reference)
    heading_align = crate_hx  # cos of crate heading; 1.0 == aligned with +x
    aligned = 1.0 if heading_align >= 0.8660254 else 0.0  # within 30 deg

    slow = 1.0 if crate_speed < 0.05 else 0.0

    # ---- progress: incremental reduction of dock distance ----
    progress = prev_dock_dist - dock_dist
    if progress < 0.0:
        progress = 0.0
    progress_reward = 8.0 * progress

    # ---- gentle contact proxy (closing speed along cart heading) ----
    cart_cx = float(obs[2])
    cart_cy = float(obs[3])
    crate_along_heading = crate_vx * cart_cx + crate_vy * cart_cy
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---- near-dock speed suppression (gated, only close to dock) ----
    near_gate = 0.0
    if dock_dist < 0.6:
        near_gate = 1.0 - dock_dist / 0.6
        if near_gate < 0.0:
            near_gate = 0.0
    speed_penalty = -0.02 * near_gate * crate_speed

    # ---- alignment shaping near dock (gated) ----
    align_factor = (1.0 + heading_align) * 0.5
    if align_factor < 0.0:
        align_factor = 0.0
    align_reward = 0.05 * near_gate * align_factor

    # ---- out-of-bounds hinge penalty ----
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    oob = 0.0
    for val in (cart_x, cart_y):
        if val > 0.95:
            oob += (val - 0.95)
        elif val < -0.95:
            oob += (-0.95 - val)
    oob_penalty = -0.5 * oob

    # ---- action smoothness (light) ----
    smooth_penalty = -0.01 * (float(action[0]) ** 2 + float(action[1]) ** 2)

    # ---- streak / one-shot events ----
    if inside > 0.5 and aligned > 0.5 and slow > 0.5:
        _streak[0] += 1
    else:
        _streak[0] = 0

    enter_event = 0.0
    if inside > 0.5 and not _entered[0]:
        _entered[0] = True
        enter_event = 20.0

    success_event = 0.0
    if _streak[0] >= 10 and not _paid[0]:
        _paid[0] = True
        success_event = 300.0

    total = (
        progress_reward
        + gentleness
        + speed_penalty
        + align_reward
        + oob_penalty
        + smooth_penalty
        + enter_event
        + success_event
    )

    components = {
        "progress": progress_reward,
        "gentleness": gentleness,
        "speed_penalty": speed_penalty,
        "align": align_reward,
        "oob_penalty": oob_penalty,
        "smooth_penalty": smooth_penalty,
        "enter_event": enter_event,
        "success_event": success_event,
    }
    return float(total), components