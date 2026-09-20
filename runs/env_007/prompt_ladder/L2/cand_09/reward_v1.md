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

    # ---------- geometry ----------
    dx = float(next_obs[12]) * 5.0      # crate-to-dock x offset (m)
    dy = float(next_obs[13]) * 4.0      # crate-to-dock y offset (m)
    dist = (dx * dx + dy * dy) ** 0.5

    odx = float(obs[12]) * 5.0
    ody = float(obs[13]) * 4.0
    old_dist = (odx * odx + ody * ody) ** 0.5

    # ---------- completion condition (explicit from obs) ----------
    inside = 1.0 if (abs(float(next_obs[12])) <= 0.024 and abs(float(next_obs[13])) <= 0.030) else 0.0

    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    crate_ang = 0.0
    ccos = float(next_obs[10])
    csin = float(next_obs[11])
    crate_ang = (csin * csin) ** 0.5
    if ccos < 0.0:
        crate_ang = 1.0
    aligned = 1.0 if crate_ang < 0.5 else 0.0   # within ~30 deg

    slow = 1.0 if crate_speed < 0.05 else 0.0

    complete_state = 1.0 if (inside > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0

    if complete_state > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- success event (one-shot) ----------
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["success_event"] = success_event

    # ---------- first entry into dock (one-shot) ----------
    first_entry = 0.0
    if inside > 0.5 and aligned > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        first_entry = 20.0
    components["first_entry"] = first_entry

    # ---------- if in complete state: zero everything else ----------
    if complete_state > 0.5:
        total = success_event + first_entry
        return (float(total), components)

    # ---------- progress (incremental delta only) ----------
    progress = 0.0
    if old_dist > 1e-6:
        delta = old_dist - dist
        if delta > 0.0:
            progress = 2.0 * delta
            if progress > 1.0:
                progress = 1.0
    components["crate_to_dock_progress"] = progress

    # ---------- gentleness (soft contact penalty) ----------
    crate_vx_h = float(next_obs[8]) * 3.0
    crate_vy_h = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx_h * float(obs[2]) + crate_vy_h * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -2.0 * contact * closing
    components["gentleness"] = gentleness

    # ---------- crate speed penalty near dock ----------
    speed_pen = 0.0
    if dist < 1.5:
        near_gate = 1.0 - dist / 1.5
        if near_gate < 0.0:
            near_gate = 0.0
        if crate_speed > 0.05:
            speed_pen = -1.5 * near_gate * (crate_speed - 0.05)
    components["crate_speed_penalty_near_dock"] = speed_pen

    # ---------- out of bounds guard ----------
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    oob = 0.0
    ax = cart_x
    if ax < 0.0:
        ax = -ax
    ay = cart_y
    if ay < 0.0:
        ay = -ay
    if ax > 0.95:
        oob = oob - 8.0 * (ax - 0.95)
    if ay > 0.95:
        oob = oob - 8.0 * (ay - 0.95)
    components["out_of_bounds_penalty"] = oob

    # ---------- action smoothness (light) ----------
    smooth = -0.02 * (float(action[0]) * float(action[0]) + float(action[1]) * float(action[1]))
    components["action_smoothness"] = smooth

    total = (
        components["success_event"]
        + components["first_entry"]
        + components["crate_to_dock_progress"]
        + components["gentleness"]
        + components["crate_speed_penalty_near_dock"]
        + components["out_of_bounds_penalty"]
        + components["action_smoothness"]
    )
    return (float(total), components)
```