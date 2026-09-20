# Response Record

```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_DIST = [None]

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- episode boundary detection via time_fraction ----
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = None
    _PREV_T[0] = t

    # ---- dock geometry (from env facts) ----
    # |obs[12]| <= 0.024 and |obs[13]| <= 0.030 => crate fully inside dock
    # obs[12], obs[13] are normalized offsets; half-width/half-height scaling.
    HALF_W = 5.0
    HALF_H = 4.0
    dx_m = next_obs[12] * HALF_W
    dy_m = next_obs[13] * HALF_H
    dist_m = (dx_m * dx_m + dy_m * dy_m) ** 0.5

    inside = 1.0 if (abs(next_obs[12]) <= 0.024 and abs(next_obs[13]) <= 0.030) else 0.0

    # ---- crate orientation error ----
    crate_cos = next_obs[10]
    crate_sin = next_obs[11]
    # dock alignment: crate heading should be aligned with dock axis (assume dock aligned with -x/+x axis => cos ~ +-1)
    # Use |cos| closeness: error = 1 - |cos(theta)|
    align_err = 1.0 - abs(crate_cos)
    # convert to angle approx: err_angle ~ acos(|cos|)
    # approximate via 1 - |cos| as continuous proxy (0 when aligned, up to 1 when perpendicular)

    # ---- crate speed ----
    cvx = next_obs[8] * 3.0
    cvy = next_obs[9] * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---- progress: improvement_delta on distance to dock ----
    if _PREV_DIST[0] is None:
        prev_d = dist_m
    else:
        prev_d = _PREV_DIST[0]
    progress = prev_d - dist_m  # positive when getting closer
    _PREV_DIST[0] = dist_m

    # ---- component: crate_to_dock_progress (delta form, avoids hover farming) ----
    progress_reward = 2.0 * progress  # per-step improvement in meters

    # ---- component: dock entry one-time bonus ----
    entry_bonus = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        entry_bonus = 15.0

    # ---- component: docking quality shaping (only when near dock) ----
    # gate: only active when crate is reasonably close to dock (within ~1.0 m)
    near_gate = 0.0
    if dist_m < 1.0:
        near_gate = 1.0 - dist_m  # 1 at center, 0 at 1m
    # orientation alignment shaping near dock
    align_reward = 0.0
    if near_gate > 0.0:
        align_reward = 3.0 * near_gate * (1.0 - align_err)

    # ---- component: crate_speed_penalty_near_dock (hinge, only when near) ----
    speed_pen = 0.0
    if dist_m < 0.6:
        # penalize speed above small threshold only when close to dock
        excess = crate_speed - 0.05
        if excess > 0.0:
            speed_pen = -2.0 * excess * near_gate

    # ---- component: soft_contact_penalty (gentleness) ----
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---- component: out_of_bounds soft hinge ----
    # cart position obs[0], obs[1] in [-2, 2] bounds; penalize near edges
    oob_pen = 0.0
    ax = abs(next_obs[0])
    ay = abs(next_obs[1])
    if ax > 1.6:
        oob_pen -= 1.0 * (ax - 1.6)
    if ay > 1.6:
        oob_pen -= 1.0 * (ay - 1.6)
    # crate far from dock but also from cart could indicate lost crate; skip.

    # ---- completion streak ----
    speed_ok = crate_speed < 0.05
    align_ok = align_err < (1.0 - 0.8660254)  # cos(30deg) ~ 0.866
    if inside > 0.5 and speed_ok and align_ok:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    components = {
        "crate_to_dock_progress": progress_reward,
        "dock_entry_bonus": entry_bonus,
        "docking_alignment": align_reward,
        "crate_speed_near_dock": speed_pen,
        "soft_contact_penalty": gentleness,
        "out_of_bounds_penalty": oob_pen,
        "success_event": success_event,
    }

    total = (progress_reward + entry_bonus + align_reward + speed_pen
             + gentleness + oob_pen + success_event)
    return float(total), components
```
